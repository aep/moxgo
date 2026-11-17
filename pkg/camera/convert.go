package camera

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
)

// fourcc builds a V4L2 pixel format code.
func fourcc(a, b, c, d byte) uint32 {
	return uint32(a) | uint32(b)<<8 | uint32(c)<<16 | uint32(d)<<24
}

var (
	pixFmtMJPEG   = fourcc('M', 'J', 'P', 'G')
	pixFmtJPEG    = fourcc('J', 'P', 'E', 'G')
	pixFmtYUYV    = fourcc('Y', 'U', 'Y', 'V')
	pixFmtUYVY    = fourcc('U', 'Y', 'V', 'Y')
	pixFmtYVYU    = fourcc('Y', 'V', 'Y', 'U')
	pixFmtNV12    = fourcc('N', 'V', '1', '2')
	pixFmtNV21    = fourcc('N', 'V', '2', '1')
	pixFmtNV12M   = fourcc('N', 'M', '1', '2')
	pixFmtNV21M   = fourcc('N', 'M', '2', '1')
	pixFmtYUV420  = fourcc('Y', 'U', '1', '2')
	pixFmtYVU420  = fourcc('Y', 'V', '1', '2')
	pixFmtYUV420M = fourcc('Y', 'M', '1', '2')
	pixFmtGREY    = fourcc('G', 'R', 'E', 'Y')
)

// preferredFormats lists the pixel formats moxgo can decode, most preferred
// first. MJPEG comes first because it keeps USB bandwidth low at higher
// resolutions; the raw YUV formats cover SoC/ISP pipelines.
var preferredFormats = []uint32{
	pixFmtMJPEG, pixFmtJPEG,
	pixFmtYUYV, pixFmtUYVY, pixFmtYVYU,
	pixFmtNV12, pixFmtNV21, pixFmtNV12M, pixFmtNV21M,
	pixFmtYUV420, pixFmtYVU420, pixFmtYUV420M,
	pixFmtGREY,
}

func fourccString(f uint32) string {
	b := []byte{byte(f), byte(f >> 8), byte(f >> 16), byte(f >> 24)}
	for i, c := range b {
		if c < 0x20 || c > 0x7e {
			b[i] = '?'
		}
	}
	return string(b)
}

// rawFrame is one captured frame: one byte slice per memory plane, plus the
// bytes-per-line of each plane as reported by the driver (0 = tightly packed).
type rawFrame struct {
	format  uint32
	width   int
	height  int
	planes  [][]byte
	strides []int
}

func (f *rawFrame) stride(i, minimum int) int {
	if i < len(f.strides) && f.strides[i] >= minimum {
		return f.strides[i]
	}
	return minimum
}

// decodeFrame converts a raw frame into an image. YUV formats are returned as
// *image.YCbCr (no RGB conversion), which image/jpeg encodes directly.
func decodeFrame(f *rawFrame) (image.Image, error) {
	if len(f.planes) == 0 {
		return nil, fmt.Errorf("camera: frame has no planes")
	}
	w, h := f.width, f.height
	switch f.format {
	case pixFmtMJPEG, pixFmtJPEG:
		return jpeg.Decode(bytes.NewReader(f.planes[0]))

	case pixFmtYUYV:
		return packed422(f.planes[0], w, h, f.stride(0, (w+1)/2*4), 0, 1, 2, 3)
	case pixFmtUYVY:
		return packed422(f.planes[0], w, h, f.stride(0, (w+1)/2*4), 1, 0, 3, 2)
	case pixFmtYVYU:
		return packed422(f.planes[0], w, h, f.stride(0, (w+1)/2*4), 0, 3, 2, 1)

	case pixFmtNV12, pixFmtNV21:
		ys := f.stride(0, w)
		y := f.planes[0]
		if len(y) < ys*h {
			return nil, errShort(f, 0)
		}
		return semiPlanar420(y[:ys*h], ys, y[ys*h:], ys, w, h, f.format == pixFmtNV21)
	case pixFmtNV12M, pixFmtNV21M:
		if len(f.planes) < 2 {
			return nil, fmt.Errorf("camera: %s needs 2 planes, got %d", fourccString(f.format), len(f.planes))
		}
		ys := f.stride(0, w)
		return semiPlanar420(f.planes[0], ys, f.planes[1], f.stride(1, ys), w, h, f.format == pixFmtNV21M)

	case pixFmtYUV420, pixFmtYVU420:
		ys := f.stride(0, w)
		cs := (ys + 1) / 2
		ch := (h + 1) / 2
		p := f.planes[0]
		ySize, cSize := ys*h, cs*ch
		if len(p) < ySize+2*cSize {
			return nil, errShort(f, 0)
		}
		u, v := p[ySize:ySize+cSize], p[ySize+cSize:ySize+2*cSize]
		if f.format == pixFmtYVU420 {
			u, v = v, u
		}
		return planar420(p[:ySize], ys, u, v, cs, w, h)
	case pixFmtYUV420M:
		if len(f.planes) < 3 {
			return nil, fmt.Errorf("camera: YM12 needs 3 planes, got %d", len(f.planes))
		}
		ys := f.stride(0, w)
		return planar420(f.planes[0], ys, f.planes[1], f.planes[2], f.stride(1, (w+1)/2), w, h)

	case pixFmtGREY:
		s := f.stride(0, w)
		if len(f.planes[0]) < s*(h-1)+w {
			return nil, errShort(f, 0)
		}
		img := image.NewGray(image.Rect(0, 0, w, h))
		for row := range h {
			copy(img.Pix[row*w:(row+1)*w], f.planes[0][row*s:])
		}
		return img, nil
	}
	return nil, fmt.Errorf("camera: unsupported pixel format %s", fourccString(f.format))
}

func errShort(f *rawFrame, plane int) error {
	return fmt.Errorf("camera: short %s frame (plane %d: %d bytes for %dx%d)",
		fourccString(f.format), plane, len(f.planes[plane]), f.width, f.height)
}

// packed422 converts packed 4:2:2 (YUYV and byte-order variants) given the
// byte offsets of Y0, U, Y1, V within each 4-byte macropixel.
func packed422(data []byte, w, h, stride, y0, u, y1, v int) (image.Image, error) {
	if len(data) < stride*(h-1)+(w+1)/2*4 {
		return nil, fmt.Errorf("camera: short packed 4:2:2 frame (%d bytes for %dx%d)", len(data), w, h)
	}
	img := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio422)
	for row := range h {
		src := data[row*stride:]
		yd := img.Y[row*img.YStride:]
		cb := img.Cb[row*img.CStride:]
		cr := img.Cr[row*img.CStride:]
		for x := 0; x+1 < w; x += 2 {
			m := src[x*2:]
			yd[x] = m[y0]
			yd[x+1] = m[y1]
			cb[x/2] = m[u]
			cr[x/2] = m[v]
		}
		if w%2 == 1 {
			m := src[(w-1)*2:]
			yd[w-1] = m[y0]
			cb[w/2] = m[u]
			cr[w/2] = m[v]
		}
	}
	return img, nil
}

// semiPlanar420 converts NV12 (interleaved CbCr) or NV21 (interleaved CrCb).
func semiPlanar420(y []byte, yStride int, uv []byte, uvStride, w, h int, swap bool) (image.Image, error) {
	cw, ch := (w+1)/2, (h+1)/2
	if len(y) < yStride*(h-1)+w || len(uv) < uvStride*(ch-1)+cw*2 {
		return nil, fmt.Errorf("camera: short 4:2:0 semi-planar frame (y=%d uv=%d bytes for %dx%d)", len(y), len(uv), w, h)
	}
	img := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio420)
	for row := range h {
		copy(img.Y[row*img.YStride:row*img.YStride+w], y[row*yStride:])
	}
	ci, ri := 0, 1
	if swap {
		ci, ri = 1, 0
	}
	for row := range ch {
		src := uv[row*uvStride:]
		cb := img.Cb[row*img.CStride:]
		cr := img.Cr[row*img.CStride:]
		for x := range cw {
			cb[x] = src[x*2+ci]
			cr[x] = src[x*2+ri]
		}
	}
	return img, nil
}

// planar420 converts fully planar 4:2:0 (I420).
func planar420(y []byte, yStride int, u, v []byte, cStride, w, h int) (image.Image, error) {
	cw, ch := (w+1)/2, (h+1)/2
	need := cStride*(ch-1) + cw
	if len(y) < yStride*(h-1)+w || len(u) < need || len(v) < need {
		return nil, fmt.Errorf("camera: short 4:2:0 planar frame (%dx%d)", w, h)
	}
	img := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio420)
	for row := range h {
		copy(img.Y[row*img.YStride:row*img.YStride+w], y[row*yStride:])
	}
	for row := range ch {
		copy(img.Cb[row*img.CStride:row*img.CStride+cw], u[row*cStride:])
		copy(img.Cr[row*img.CStride:row*img.CStride+cw], v[row*cStride:])
	}
	return img, nil
}
