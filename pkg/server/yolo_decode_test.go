package server

import (
	"testing"
)

// yoloOutput is a [1, 4+classes, anchors] output with the given anchors
// set: box cx, cy, w, h and a score per class; the rest score nothing.
func yoloOutput(anchors, classes int, set map[int][]float32) ([]float32, []int64) {
	data := make([]float32, (4+classes)*anchors)
	for a, v := range set {
		for ch, x := range v {
			data[ch*anchors+a] = x
		}
	}
	return data, []int64{1, int64(4 + classes), int64(anchors)}
}

// Each anchor counts once, with its best class; overlapping boxes of a class
// are suppressed, of different classes not; the result runs by confidence.
// A scratch decodes the next output the same as a fresh one.
func TestYoloDecode(t *testing.T) {
	data, shape := yoloOutput(100, 3, map[int][]float32{
		5:  {50, 50, 20, 20, 0.9, 0.3, 0},  // class 0
		6:  {51, 50, 20, 20, 0.8, 0, 0},    // class 0, overlaps 5: suppressed
		7:  {51, 50, 20, 20, 0, 0, 0.7},    // class 2, overlaps 5: kept
		40: {10, 10, 4, 4, 0.2, 0.95, 0.5}, // class 1 is its best
		41: {80, 80, 4, 4, 0.1, 0.1, 0.1},  // below the threshold
	})
	want := []struct {
		anchor, class int
		conf          float32
	}{{40, 1, 0.95}, {5, 0, 0.9}, {7, 2, 0.7}}
	var s yoloScratch
	for round := range 3 {
		got := s.decode(data, shape, 3)
		if len(got) != len(want) {
			t.Fatalf("round %d: %d detections, want %d: %+v", round, len(got), len(want), got)
		}
		for i, w := range want {
			if got[i].anchor != w.anchor || got[i].classIdx != w.class || got[i].conf != w.conf {
				t.Fatalf("round %d: detection %d is %+v, want %+v", round, i, got[i], w)
			}
		}
		if g := got[1]; g.x1 != 40 || g.y1 != 40 || g.x2 != 60 || g.y2 != 60 {
			t.Fatalf("box of anchor 5: %+v", g)
		}
	}
	for _, b := range s.best {
		if b != 0 {
			t.Fatal("the anchor table is not left clean")
		}
	}
	if n := testing.AllocsPerRun(10, func() { s.decode(data, shape, 3) }); n > 0 {
		t.Fatalf("%v allocations per decode with a warm scratch", n)
	}
}
