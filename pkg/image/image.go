// Package image decodes images and prepares them as CHW float32 tensors
// for direct use with pkg/onnx.
package image

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"

	_ "golang.org/x/image/bmp"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

// Tensor holds a preprocessed image ready for ONNX inference.
// Data is in NCHW layout, normalized to [0,1].
type Tensor struct {
	Data  []float32
	Shape []int64 // [1, C, H, W]

	// Letterbox transform info for mapping output coordinates back to original.
	Scale float64
	PadX  int
	PadY  int
	SrcW  int
	SrcH  int
}

// Decode reads an image from r, resizes it with letterboxing to width×height,
// normalizes pixels to [0,1], and returns a CHW float32 tensor.
func Decode(r io.Reader, width, height int) (*Tensor, error) {
	src, _, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("image: decode: %w", err)
	}
	return prepare(src, width, height), nil
}

// FromImage converts an already-decoded image to a tensor.
func FromImage(src image.Image, width, height int) *Tensor {
	return prepare(src, width, height)
}

// DecodeUint8CHW reads an image and resizes it (plain stretch — no aspect
// preservation, no padding) to width×height, returning a raw uint8 CHW [3,H,W]
// buffer plus the original dimensions. This matches SAM 3's image-encoder
// preprocessing: bilinear resize, RGB, no normalization (the model bakes it in).
func DecodeUint8CHW(r io.Reader, width, height int) (data []uint8, srcW, srcH int, err error) {
	src, _, e := image.Decode(r)
	if e != nil {
		return nil, 0, 0, fmt.Errorf("image: decode: %w", e)
	}
	d, w, h := prepareUint8CHW(src, width, height)
	return d, w, h, nil
}

// FromImageUint8CHW is DecodeUint8CHW for an already-decoded image (e.g. a
// camera frame).
func FromImageUint8CHW(src image.Image, width, height int) (data []uint8, srcW, srcH int) {
	return prepareUint8CHW(src, width, height)
}

func prepareUint8CHW(src image.Image, width, height int) ([]uint8, int, int) {
	bounds := src.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()

	// Stretch to the target square (bilinear, closest to cv2.INTER_LINEAR).
	resized := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.BiLinear.Scale(resized, resized.Bounds(), src, bounds, draw.Over, nil)

	chSize := width * height
	data := make([]uint8, 3*chSize)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			si := y*resized.Stride + x*4
			data[0*chSize+y*width+x] = resized.Pix[si]   // R
			data[1*chSize+y*width+x] = resized.Pix[si+1] // G
			data[2*chSize+y*width+x] = resized.Pix[si+2] // B
		}
	}
	return data, srcW, srcH
}

func prepare(src image.Image, width, height int) *Tensor {
	bounds := src.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()

	// Compute letterbox scale and padding.
	scaleW := float64(width) / float64(srcW)
	scaleH := float64(height) / float64(srcH)
	scale := scaleW
	if scaleH < scale {
		scale = scaleH
	}
	newW := int(float64(srcW) * scale)
	newH := int(float64(srcH) * scale)
	padX := (width - newW) / 2
	padY := (height - newH) / 2

	// Resize source to newW×newH using Catmull-Rom (high quality).
	resized := image.NewRGBA(image.Rect(0, 0, newW, newH))
	draw.CatmullRom.Scale(resized, resized.Bounds(), src, bounds, draw.Over, nil)

	// Build CHW float32 tensor with letterbox padding (pad value = 114/255).
	const padVal = 114.0 / 255.0
	n := width * height * 3
	data := make([]float32, n)

	// Fill with pad value.
	for i := range data {
		data[i] = padVal
	}

	// Write resized pixels into the padded region, converting HWC→CHW + normalizing.
	chSize := width * height
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			srcIdx := y*resized.Stride + x*4
			r := float32(resized.Pix[srcIdx]) / 255.0
			g := float32(resized.Pix[srcIdx+1]) / 255.0
			b := float32(resized.Pix[srcIdx+2]) / 255.0

			dy := y + padY
			dx := x + padX
			data[0*chSize+dy*width+dx] = r
			data[1*chSize+dy*width+dx] = g
			data[2*chSize+dy*width+dx] = b
		}
	}

	return &Tensor{
		Data:  data,
		Shape: []int64{1, 3, int64(height), int64(width)},
		Scale: scale,
		PadX:  padX,
		PadY:  padY,
		SrcW:  srcW,
		SrcH:  srcH,
	}
}
