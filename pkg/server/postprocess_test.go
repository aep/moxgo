package server

import (
	"testing"

	goimage "github.com/aep/moxgo/pkg/image"
)

// A 32×32 image letterboxed into a 16×16 model input, one class, one mask
// prototype (4×4) that is positive on its left half: a detection over the
// whole image has the image's left half as its mask.
func TestSegmentOutput(t *testing.T) {
	const nDets = 2
	// [1, 4+1+1, 2]: rows cx, cy, w, h, class score, mask coefficient.
	data := []float32{
		8, 3, // cx
		8, 3, // cy
		16, 2, // w
		16, 2, // h
		0.9, 0.1, // score: the second is below the threshold
		1, 1, // coefficient
	}
	shape := []int64{1, 6, nDets}
	protos := make([]float32, 16)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			protos[y*4+x] = 1
			if x >= 2 {
				protos[y*4+x] = -1
			}
		}
	}
	img := &goimage.Tensor{Shape: []int64{1, 3, 16, 16}, Scale: 0.5, SrcW: 32, SrcH: 32}

	dets := SegmentOutput(data, shape, protos, []int64{1, 1, 4, 4}, nil, img)
	if len(dets.Items) != 1 {
		t.Fatalf("got %d detections, want 1", len(dets.Items))
	}
	d := dets.Items[0]
	if d.X1 != 0 || d.Y1 != 0 || d.X2 != 32 || d.Y2 != 32 {
		t.Fatalf("box %v %v %v %v, want 0 0 32 32", d.X1, d.Y1, d.X2, d.Y2)
	}
	m := d.Mask
	if m == nil || m.X != 0 || m.Y != 0 || m.Width != 32 || m.Height != 32 {
		t.Fatalf("mask %+v, want 32×32 at 0,0", m)
	}
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			i := y*32 + x
			got := m.Bits[i/8]&(0x80>>(i%8)) != 0
			if want := x < 16; got != want {
				t.Fatalf("mask at %d,%d = %v, want %v", x, y, got, want)
			}
		}
	}
}
