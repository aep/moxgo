package image

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func i420(w, h int, fn func(x, y int) (yv, u, v uint8)) *Planes {
	cw, ch := (w+1)/2, (h+1)/2
	p := &Planes{Width: w, Height: h, Y: make([]byte, w*h), YStride: w, U: make([]byte, cw*ch), V: make([]byte, cw*ch), UVStride: cw}
	for y := range h {
		for x := range w {
			yv, u, v := fn(x, y)
			p.Y[y*w+x] = yv
			if x&1 == 0 && y&1 == 0 {
				p.U[y/2*cw+x/2], p.V[y/2*cw+x/2] = u, v
			}
		}
	}
	return p
}

func nv12From(p *Planes) *Planes {
	cw, ch := (p.Width+1)/2, (p.Height+1)/2
	uv := make([]byte, 2*cw*ch)
	for i := range cw * ch {
		uv[2*i], uv[2*i+1] = p.U[i], p.V[i]
	}
	q := *p
	q.U, q.V, q.UVStride = uv, nil, 2*cw
	return &q
}

func gradient(x, y int) (uint8, uint8, uint8) {
	return uint8(40 + (x*150/2560 + y*50/1920)), uint8(110 + y*30/1920), uint8(150 - x*30/2560)
}

func TestLetterboxAndPad(t *testing.T) {
	const W, H = 64, 64
	p := NewPreprocessor(W, H, nil, nil)
	dst := make([]float32, 3*W*H)
	src := i420(128, 96, func(int, int) (uint8, uint8, uint8) { return 128, 128, 128 })
	lb := p.Process(dst, src)
	if lb.ContentW != 64 || lb.ContentH != 48 || lb.PadX != 0 || lb.PadY != 8 || lb.Scale != 0.5 {
		t.Fatalf("letterbox %+v", lb)
	}
	if v := dst[0]; math.Abs(float64(v-padValue)) > 1e-5 {
		t.Fatalf("pad %v", v)
	}
	if v := dst[8*W+10]; math.Abs(float64(v)-128.0/255) > 0.01 {
		t.Fatalf("content %v", v)
	}
	if v := dst[2*W*H+56*W+10]; math.Abs(float64(v-padValue)) > 1e-5 {
		t.Fatalf("bottom pad %v", v)
	}
	// A narrower frame on the same buffer gets its own border.
	lb = p.Process(dst, i420(32, 64, func(int, int) (uint8, uint8, uint8) { return 128, 128, 128 }))
	if lb.PadX != 16 || dst[5] != float32(padValue) {
		t.Fatalf("pad after geometry change: %+v %v", lb, dst[5])
	}
}

// NV12 and I420 of the same picture give the same tensor.
func TestNV12MatchesI420(t *testing.T) {
	src := i420(320, 180, gradient)
	p := NewPreprocessor(160, 160, nil, nil)
	a, b := make([]float32, 3*160*160), make([]float32, 3*160*160)
	p.Process(a, src)
	p.Process(b, nv12From(src))
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("differ at %d: %v %v", i, a[i], b[i])
		}
	}
}

// The fast path stays close to the reference path over several scales.
func TestMatchesReferencePath(t *testing.T) {
	for _, size := range [][2]int{{640, 480}, {1280, 720}, {1920, 1080}, {2560, 1920}, {320, 240}, {800, 600}} {
		w, h := size[0], size[1]
		src := i420(w, h, gradient)
		img := &image.YCbCr{Y: src.Y, Cb: src.U, Cr: src.V, YStride: src.YStride, CStride: src.UVStride,
			SubsampleRatio: image.YCbCrSubsampleRatio420, Rect: image.Rect(0, 0, w, h)}
		ref := FromImage(img, 640, 640)
		p := NewPreprocessor(640, 640, nil, nil)
		got := make([]float32, 3*640*640)
		lb := p.Process(got, src)
		if lb.ContentW != ref.SrcW && float64(lb.Scale) != ref.Scale {
			t.Fatalf("%dx%d: letterbox %+v vs %+v", w, h, lb, ref)
		}
		var sum float64
		maxDiff := 0.0
		for i := range got {
			d := math.Abs(float64(got[i] - ref.Data[i]))
			sum += d
			maxDiff = max(maxDiff, d)
		}
		avg := sum / float64(len(got))
		if avg > 0.01 || maxDiff > 0.08 {
			t.Errorf("%dx%d: avg diff %.4f max %.4f", w, h, avg, maxDiff)
		}
	}
}

func TestNormalizationAndRange(t *testing.T) {
	mean, std := []float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225}
	p := NewPreprocessor(8, 8, mean, std)
	dst := make([]float32, 3*8*8)
	// Limited-range white is 1.0 before normalization, full-range 235 is not.
	src := i420(16, 16, func(int, int) (uint8, uint8, uint8) { return 235, 128, 128 })
	src.Limited = true
	p.Process(dst, src)
	want := (1 - mean[0]) / std[0]
	if math.Abs(float64(dst[0]-want)) > 0.01 {
		t.Fatalf("limited white %v want %v", dst[0], want)
	}
	src.Limited = false
	p.Process(dst, src)
	want = (235.0/255 - mean[0]) / std[0]
	if math.Abs(float64(dst[0]-want)) > 0.01 {
		t.Fatalf("full 235 gave %v want %v", dst[0], want)
	}
}

func TestProcessDoesNotAllocate(t *testing.T) {
	src := nv12From(i420(1280, 720, gradient))
	p := NewPreprocessor(640, 640, nil, nil)
	dst := make([]float32, 3*640*640)
	p.Process(dst, src)
	if n := testing.AllocsPerRun(10, func() { p.Process(dst, src) }); n > 0 {
		t.Fatalf("%v allocations per call", n)
	}
}

func BenchmarkPreprocess(b *testing.B) {
	for _, size := range [][2]int{{640, 480}, {1280, 720}, {1920, 1080}, {2560, 1920}} {
		src := nv12From(i420(size[0], size[1], gradient))
		b.Run(string(rune('0'+size[0]/640))+"x", func(b *testing.B) {
			p := NewPreprocessor(640, 640, nil, nil)
			dst := make([]float32, 3*640*640)
			b.SetBytes(int64(size[0] * size[1]))
			for b.Loop() {
				p.Process(dst, src)
			}
		})
	}
}

// BenchmarkPreprocessInvalidated is a picture already at the model's
// width (a 720p frame halved), with dst invalidated before each call, as
// an Instance does: the slot may have held another picture.
func BenchmarkPreprocessInvalidated(b *testing.B) {
	img := image.NewYCbCr(image.Rect(0, 0, 640, 360), image.YCbCrSubsampleRatio420)
	src, _ := PlanesFromYCbCr(img)
	p := NewPreprocessor(640, 640, nil, nil)
	dst := make([]float32, 3*640*640)
	b.SetBytes(640 * 360)
	for b.Loop() {
		p.Invalidate()
		p.Process(dst, &src)
	}
}

func BenchmarkReferencePath(b *testing.B) {
	img := image.NewYCbCr(image.Rect(0, 0, 1280, 720), image.YCbCrSubsampleRatio420)
	for i := range img.Y {
		img.Y[i] = uint8(i)
	}
	for b.Loop() {
		FromImage(img, 640, 640)
	}
}

var _ = color.White
