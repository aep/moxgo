package image

import (
	"image/color"
	"testing"
)

func TestYCbCr420(t *testing.T) {
	// 5×3: chroma planes are 3×2, odd sizes round up
	y, c := make([]byte, 15), make([]byte, 6)
	if _, err := YCbCr420(5, 3, y, c, c); err != nil {
		t.Fatal(err)
	}
	for name, planes := range map[string][3][]byte{
		"short y":  {y[:14], c, c},
		"short cb": {y, c[:5], c},
		"long cr":  {y, c, append(c, 0)},
	} {
		if _, err := YCbCr420(5, 3, planes[0], planes[1], planes[2]); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if _, err := YCbCr420(0, 3, nil, nil, nil); err == nil {
		t.Error("empty image: no error")
	}
}

// A picture as wide as the model input is letterboxed without resampling:
// its pixels are the YCbCr colors converted, and the rows around it padding.
func TestLetterboxAtScaleOne(t *testing.T) {
	const w, h = 8, 4
	y, cb, cr := make([]byte, w*h), make([]byte, w/2*h/2), make([]byte, w/2*h/2)
	for i := range y {
		y[i] = byte(16 + 7*i)
	}
	for i := range cb {
		cb[i], cr[i] = byte(90+11*i), byte(200-13*i)
	}
	img, err := YCbCr420(w, h, y, cb, cr)
	if err != nil {
		t.Fatal(err)
	}
	ts := FromImage(img, w, w) // 8×4 into 8×8: 2 rows of padding above and below
	if ts.Scale != 1 || ts.PadX != 0 || ts.PadY != 2 {
		t.Fatalf("scale %v pad %d,%d", ts.Scale, ts.PadX, ts.PadY)
	}
	plane := w * w
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			ci := (py/2)*(w/2) + px/2
			r, g, b := color.YCbCrToRGB(y[py*w+px], cb[ci], cr[ci])
			i := (py+ts.PadY)*w + px
			got := [3]float32{ts.Data[i], ts.Data[plane+i], ts.Data[2*plane+i]}
			want := [3]float32{float32(r) / 255, float32(g) / 255, float32(b) / 255}
			if got != want {
				t.Fatalf("pixel %d,%d = %v, want %v", px, py, got, want)
			}
		}
	}
	if ts.Data[0] != 114.0/255.0 {
		t.Errorf("padding = %v", ts.Data[0])
	}
}
