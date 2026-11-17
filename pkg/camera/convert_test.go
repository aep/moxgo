package camera

import (
	"bytes"
	"image"
	"image/jpeg"
	"testing"
)

// testPlanes returns deterministic Y and 4:2:0 / 4:2:2 chroma samples.
func yAt(x, y int) byte  { return byte(x*7 + y*13) }
func cbAt(x, y int) byte { return byte(x*3 + y*5 + 40) }
func crAt(x, y int) byte { return byte(x*11 + y*2 + 90) }

const pad = 16 // extra bytes per line, as ISPs commonly align strides

func checkYCbCr(t *testing.T, img image.Image, w, h int, ratio image.YCbCrSubsampleRatio) {
	t.Helper()
	yc, ok := img.(*image.YCbCr)
	if !ok {
		t.Fatalf("got %T, want *image.YCbCr", img)
	}
	if yc.SubsampleRatio != ratio || yc.Rect.Dx() != w || yc.Rect.Dy() != h {
		t.Fatalf("got %v %v, want %dx%d %v", yc.Rect, yc.SubsampleRatio, w, h, ratio)
	}
	for y := range h {
		for x := range w {
			if got := yc.Y[yc.YOffset(x, y)]; got != yAt(x, y) {
				t.Fatalf("Y(%d,%d)=%d want %d", x, y, got, yAt(x, y))
			}
			cy := y
			if ratio == image.YCbCrSubsampleRatio420 {
				cy = y / 2
			}
			co := yc.COffset(x, y)
			if got := yc.Cb[co]; got != cbAt(x/2, cy) {
				t.Fatalf("Cb(%d,%d)=%d want %d", x, y, got, cbAt(x/2, cy))
			}
			if got := yc.Cr[co]; got != crAt(x/2, cy) {
				t.Fatalf("Cr(%d,%d)=%d want %d", x, y, got, crAt(x/2, cy))
			}
		}
	}
}

func TestPacked422(t *testing.T) {
	for _, tc := range []struct {
		name         string
		fmt          uint32
		y0, u, y1, v int
	}{
		{"YUYV", pixFmtYUYV, 0, 1, 2, 3},
		{"UYVY", pixFmtUYVY, 1, 0, 3, 2},
		{"YVYU", pixFmtYVYU, 0, 3, 2, 1},
	} {
		for _, w := range []int{8, 7} {
			h := 5
			stride := (w+1)/2*4 + pad
			buf := make([]byte, stride*h)
			for y := range h {
				for mx := 0; mx < (w+1)/2; mx++ {
					m := buf[y*stride+mx*4:]
					m[tc.y0] = yAt(mx*2, y)
					m[tc.y1] = yAt(mx*2+1, y)
					m[tc.u] = cbAt(mx, y)
					m[tc.v] = crAt(mx, y)
				}
			}
			img, err := decodeFrame(&rawFrame{format: tc.fmt, width: w, height: h,
				planes: [][]byte{buf}, strides: []int{stride}})
			if err != nil {
				t.Fatalf("%s w=%d: %v", tc.name, w, err)
			}
			checkYCbCr(t, img, w, h, image.YCbCrSubsampleRatio422)
		}
	}
}

func make420(w, h, ys, cs int, interleaved, swap bool) (y, c []byte) {
	cw, ch := (w+1)/2, (h+1)/2
	y = make([]byte, ys*h)
	for r := range h {
		for x := range w {
			y[r*ys+x] = yAt(x, r)
		}
	}
	if interleaved {
		c = make([]byte, cs*ch)
		for r := range ch {
			for x := range cw {
				a, b := cbAt(x, r), crAt(x, r)
				if swap {
					a, b = b, a
				}
				c[r*cs+x*2], c[r*cs+x*2+1] = a, b
			}
		}
		return
	}
	c = make([]byte, 2*cs*ch)
	for r := range ch {
		for x := range cw {
			u, v := cbAt(x, r), crAt(x, r)
			if swap {
				u, v = v, u
			}
			c[r*cs+x] = u
			c[cs*ch+r*cs+x] = v
		}
	}
	return
}

func TestSemiPlanar420(t *testing.T) {
	for _, w := range []int{8, 7} {
		h := 6
		ys := w + pad + w%2
		for _, swap := range []bool{false, true} {
			y, uv := make420(w, h, ys, ys, true, swap)

			single := pixFmtNV12
			multi := pixFmtNV12M
			if swap {
				single, multi = pixFmtNV21, pixFmtNV21M
			}
			img, err := decodeFrame(&rawFrame{format: single, width: w, height: h,
				planes: [][]byte{append(append([]byte{}, y...), uv...)}, strides: []int{ys}})
			if err != nil {
				t.Fatal(err)
			}
			checkYCbCr(t, img, w, h, image.YCbCrSubsampleRatio420)

			img, err = decodeFrame(&rawFrame{format: multi, width: w, height: h,
				planes: [][]byte{y, uv}, strides: []int{ys, ys}})
			if err != nil {
				t.Fatal(err)
			}
			checkYCbCr(t, img, w, h, image.YCbCrSubsampleRatio420)
		}
	}
}

func TestPlanar420(t *testing.T) {
	for _, w := range []int{8, 7} {
		h := 6
		ys := w + pad + w%2 // even stride so chroma stride is ys/2
		cs := ys / 2
		for _, swap := range []bool{false, true} {
			y, c := make420(w, h, ys, cs, false, swap)
			f := pixFmtYUV420
			if swap {
				f = pixFmtYVU420
			}
			img, err := decodeFrame(&rawFrame{format: f, width: w, height: h,
				planes: [][]byte{append(append([]byte{}, y...), c...)}, strides: []int{ys}})
			if err != nil {
				t.Fatal(err)
			}
			checkYCbCr(t, img, w, h, image.YCbCrSubsampleRatio420)
		}
		y, c := make420(w, h, ys, cs, false, false)
		half := len(c) / 2
		img, err := decodeFrame(&rawFrame{format: pixFmtYUV420M, width: w, height: h,
			planes: [][]byte{y, c[:half], c[half:]}, strides: []int{ys, cs, cs}})
		if err != nil {
			t.Fatal(err)
		}
		checkYCbCr(t, img, w, h, image.YCbCrSubsampleRatio420)
	}
}

func TestGreyAndJPEG(t *testing.T) {
	w, h := 9, 4
	s := w + pad
	buf := make([]byte, s*h)
	for y := range h {
		for x := range w {
			buf[y*s+x] = yAt(x, y)
		}
	}
	img, err := decodeFrame(&rawFrame{format: pixFmtGREY, width: w, height: h,
		planes: [][]byte{buf}, strides: []int{s}})
	if err != nil {
		t.Fatal(err)
	}
	g := img.(*image.Gray)
	for y := range h {
		for x := range w {
			if g.GrayAt(x, y).Y != yAt(x, y) {
				t.Fatalf("grey mismatch at %d,%d", x, y)
			}
		}
	}

	var jb bytes.Buffer
	if err := jpeg.Encode(&jb, img, nil); err != nil {
		t.Fatal(err)
	}
	img, err = decodeFrame(&rawFrame{format: pixFmtMJPEG, width: w, height: h, planes: [][]byte{jb.Bytes()}})
	if err != nil || img.Bounds().Dx() != w {
		t.Fatalf("mjpeg: %v %v", img, err)
	}
}

func TestShortFrames(t *testing.T) {
	for _, f := range []uint32{pixFmtYUYV, pixFmtNV12, pixFmtNV12M, pixFmtYUV420, pixFmtYUV420M, pixFmtGREY} {
		_, err := decodeFrame(&rawFrame{format: f, width: 64, height: 48,
			planes: [][]byte{make([]byte, 10), make([]byte, 10), make([]byte, 10)}})
		if err == nil {
			t.Errorf("%s: expected error for short frame", fourccString(f))
		}
	}
	if _, err := decodeFrame(&rawFrame{format: fourcc('B', 'A', '8', '1'), width: 4, height: 4,
		planes: [][]byte{make([]byte, 16)}}); err == nil {
		t.Error("expected unsupported format error")
	}
}
