package live

import (
	"math"
	"testing"

	"github.com/aep/gaal"
	goimage "github.com/aep/moxgo/pkg/image"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
)

func grayFrame(w, h int, format gaal.PixelFormat, r gaal.ColorRange) *gaal.VideoFrame {
	f := gaal.NewVideoFrame(w, h, format)
	f.Range = r
	for i := range f.Y {
		f.Y[i] = 100
	}
	for i := range f.U {
		f.U[i] = 128
	}
	for i := range f.V {
		f.V[i] = 128
	}
	return f
}

// A detection in letterboxed content coordinates maps to the frame's pixels.
func TestMappingFromLetterbox(t *testing.T) {
	// 2560×1920 into 640×640: content 640×480, pad 80 top and bottom.
	lb := goimage.ComputeLetterbox(2560, 1920, 640, 640)
	if lb.ContentW != 640 || lb.ContentH != 480 || lb.PadY != 80 || lb.PadX != 0 {
		t.Fatalf("letterbox %+v", lb)
	}
	m := NewMapping(float64(lb.ContentW), float64(lb.ContentH), 2560, 1920)
	x1, y1, x2, y2, ok := m.Box(160, 120, 320, 240, 2560, 1920)
	if !ok || x1 != 640 || y1 != 480 || x2 != 1280 || y2 != 960 {
		t.Fatalf("got %d,%d %d,%d", x1, y1, x2, y2)
	}
	// Boxes leaving the frame are clamped; one wholly outside is rejected.
	x1, y1, x2, y2, ok = m.Box(-10, -10, 700, 500, 2560, 1920)
	if !ok || x1 != 0 || y1 != 0 || x2 != 2560 || y2 != 1920 {
		t.Fatalf("clamped to %d,%d %d,%d", x1, y1, x2, y2)
	}
	if _, _, _, _, ok := m.Box(700, 10, 800, 20, 2560, 1920); ok {
		t.Fatal("box outside the frame accepted")
	}
	// Anisotropic source pixels: 1280×720 in a 640×640 model, 16:9 content.
	lb = goimage.ComputeLetterbox(1280, 720, 640, 640)
	m = NewMapping(float64(lb.ContentW), float64(lb.ContentH), 1280, 720)
	if m.SX != 2 || m.SY != 2 {
		t.Fatalf("mapping %+v for %+v", m, lb)
	}
}

// yuvToRGB is the frame-side inverse used to check drawn colors.
func yuvToRGB(y, u, v uint8, full, bt709 bool) (r, g, b float64) {
	kr, kb := 0.299, 0.114
	if bt709 {
		kr, kb = 0.2126, 0.0722
	}
	kg := 1 - kr - kb
	Y, U, V := float64(y), float64(u)-128, float64(v)-128
	cs := 1.0
	if !full {
		Y = (Y - 16) * 255 / 219
		cs = 255.0 / 224
	}
	r = Y + 2*(1-kr)*V*cs
	b = Y + 2*(1-kb)*U*cs
	g = Y - 2*(1-kb)*kb/kg*U*cs - 2*(1-kr)*kr/kg*V*cs
	return
}

func TestColorRoundTrip(t *testing.T) {
	for _, full := range []bool{true, false} {
		for _, bt709 := range []bool{true, false} {
			c := RGBColor(255, 56, 56, full, bt709)
			r, g, b := yuvToRGB(c.Y, c.U, c.V, full, bt709)
			if math.Abs(r-255) > 6 || math.Abs(g-56) > 6 || math.Abs(b-56) > 6 {
				t.Fatalf("full=%v 709=%v: got %.0f,%.0f,%.0f", full, bt709, r, g, b)
			}
		}
	}
}

func TestDrawBoxBothFormats(t *testing.T) {
	for _, format := range []gaal.PixelFormat{gaal.NV12, gaal.I420} {
		for _, rng := range []gaal.ColorRange{gaal.RangeFull, gaal.RangeLimited, gaal.RangeUnspecified} {
			f := grayFrame(320, 240, format, rng)
			var o Overlay
			items := []Item{{Label: "dog", Class: 0, Confidence: 0.5, X1: 40, Y1: 40, X2: 120, Y2: 100}}
			// Result coordinates are half the frame's.
			o.Draw(f, items, 160, 120, false)
			want := RGBColor(palette[0][0], palette[0][1], palette[0][2], rng == gaal.RangeFull, false)
			if got := f.Y[80*f.YStride+150]; got != want.Y {
				t.Errorf("format %v: top edge Y=%d want %d", format, got, want.Y)
			}
			if got := f.Y[80*f.YStride+50]; got != 100 {
				t.Errorf("left of the box touched: %d", got)
			}
			if got := f.Y[150*f.YStride+160]; got != 100 {
				t.Errorf("box inside touched: %d", got)
			}
			cRow := 40 * f.UVStride
			var u, v uint8
			if format == gaal.NV12 {
				u, v = f.U[cRow+75*2], f.U[cRow+75*2+1]
			} else {
				u, v = f.U[cRow+75], f.V[cRow+75]
			}
			if u != want.U || v != want.V {
				t.Errorf("chroma %d,%d want %d,%d", u, v, want.U, want.V)
			}
		}
	}
}

func TestDrawClippedBoxDoesNotPanic(t *testing.T) {
	f := grayFrame(64, 48, gaal.NV12, gaal.RangeLimited)
	var o Overlay
	o.Draw(f, []Item{
		{Label: "a very long label that does not fit", Confidence: 1, X1: -100, Y1: -100, X2: 1000, Y2: 1000},
		{Label: "x", X1: 60, Y1: 44, X2: 70, Y2: 60},
		{Label: "classification", Confidence: 0.9},
	}, 64, 48, false)
}

func packMask(w, h int, set func(x, y int) bool) []byte {
	b := make([]byte, (w*h+7)/8)
	for y := range h {
		for x := range w {
			if set(x, y) {
				i := y*w + x
				b[i>>3] |= 0x80 >> (i & 7)
			}
		}
	}
	return b
}

// The mask is scaled with the mapping, blends only set pixels, and its chroma
// lands on the matching chroma samples.
func TestDrawMaskScaled(t *testing.T) {
	for _, format := range []gaal.PixelFormat{gaal.NV12, gaal.I420} {
		f := grayFrame(160, 120, format, gaal.RangeFull)
		// A 8×6 mask at (4,2) in 80×60 coordinates, set on its left half.
		mk := &gomaxv1.Mask{X: 4, Y: 2, Width: 8, Height: 6, Bits: packMask(8, 6, func(x, y int) bool { return x < 4 })}
		var o Overlay
		o.drawMask(f, mk, NewMapping(80, 60, 160, 120), Color{Y: 200, U: 50, V: 200})
		// Mask pixel (0,0) covers frame (8..9, 4..5).
		if got := f.Y[4*f.YStride+8]; got != 150 {
			t.Errorf("set pixel Y=%d want 150", got)
		}
		if got := f.Y[4*f.YStride+7]; got != 100 {
			t.Errorf("left neighbour changed: %d", got)
		}
		// Mask x=4 is clear: frame x 16.
		if got := f.Y[4*f.YStride+16]; got != 100 {
			t.Errorf("clear pixel changed: %d", got)
		}
		cRow := 2 * f.UVStride
		var u, v uint8
		if format == gaal.NV12 {
			u, v = f.U[cRow+4*2], f.U[cRow+4*2+1]
		} else {
			u, v = f.U[cRow+4], f.V[cRow+4]
		}
		if u != 89 || v != 164 {
			t.Errorf("chroma %d,%d want 89,164", u, v)
		}
		// A malformed mask is ignored.
		o.drawMask(f, &gomaxv1.Mask{Width: 100, Height: 100, Bits: []byte{1}}, Mapping{SX: 1, SY: 1}, Color{Y: 255})
	}
}
