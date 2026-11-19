package live

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/aep/gaal"
	"github.com/aep/gaal/display"
	"github.com/aep/moxgo/pkg/camera"
)

// grayStream is a steady mid-gray source.
type grayStream struct {
	pool gaal.FramePool
	end  time.Time
}

func (s *grayStream) Next(time.Duration) (*gaal.VideoFrame, camera.FrameTiming, error) {
	if time.Now().After(s.end) {
		return nil, camera.FrameTiming{}, errDone
	}
	time.Sleep(20 * time.Millisecond)
	f := s.pool.Get(640, 480, gaal.NV12)
	f.Range = gaal.RangeFull
	for i := range f.Y {
		f.Y[i] = 100
	}
	for i := range f.U {
		f.U[i] = 128
	}
	now := time.Now()
	return f, camera.FrameTiming{Captured: now, Received: now, Sent: now, Decoded: now}, nil
}

func (s *grayStream) Name() string               { return "gray" }
func (s *grayStream) Matrix() camera.ColorMatrix { return camera.MatrixBT601 }
func (s *grayStream) Close() error               { return nil }

var errDone = os.ErrDeadlineExceeded

type fixedBox struct{}

func (fixedBox) Prepare(*gaal.VideoFrame) error { return nil }
func (fixedBox) Infer(context.Context) (*Result, Times, error) {
	// A quarter of the frame, in 320×240 coordinates.
	return &Result{Items: []Item{{Label: "", X1: 80, Y1: 60, X2: 240, Y2: 180}}, CoordW: 320, CoordH: 240}, Times{}, nil
}
func (fixedBox) Close() {}

// TestRealDisplay opens a real window (set MOXGO_TEST_DISPLAY=1): the box must
// appear where the mapping puts it, in the box color, on a neutral background.
// Only numbers from a synthetic picture are inspected.
func TestRealDisplay(t *testing.T) {
	if os.Getenv("MOXGO_TEST_DISPLAY") == "" {
		t.Skip("MOXGO_TEST_DISPLAY is not set")
	}
	d, err := display.Open(display.Options{Title: "moxgo test", Width: 640, Height: 480, Matrix: display.BT601, Backend: os.Getenv("MOXGO_TEST_DISPLAY_BACKEND")})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	src := &grayStream{end: time.Now().Add(2 * time.Second)}
	p := New(Options{Source: src, Detector: fixedBox{}, Display: d, StartupSkip: -1, NoRedraw: true})
	done := make(chan error, 1)
	go func() { done <- p.Run(context.Background()) }()
	time.Sleep(1200 * time.Millisecond)
	img, source, err := d.Readback()
	if err != nil {
		t.Skipf("no readback on %s: %v", d.Info().Backend, err)
	}
	<-done
	b := img.Bounds()
	t.Logf("backend %s path %s readback %s %dx%d", d.Info().Backend, d.Info().Path, source, b.Dx(), b.Dy())
	minX, minY, maxX, maxY := 1<<30, 1<<30, -1, -1
	reds, neutral, other := 0, 0, 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := img.RGBAAt(x, y)
			switch {
			case c.R > 200 && c.G < 110 && c.B < 110:
				reds++
				minX, minY, maxX, maxY = min(minX, x), min(minY, y), max(maxX, x+1), max(maxY, y+1)
			case abs(int(c.R)-int(c.G)) < 12 && abs(int(c.G)-int(c.B)) < 12:
				neutral++
			default:
				other++
			}
		}
	}
	if reds == 0 {
		t.Fatalf("no box pixels (neutral %d, other %d)", neutral, other)
	}
	// The picture is fitted into the window, keeping its 4:3 shape.
	bx, by, w, h := float64(b.Min.X), float64(b.Min.Y), float64(b.Dx()), float64(b.Dy())
	if w*3 > h*4 {
		bx, w = bx+(w-h*4/3)/2, h*4/3
	} else {
		by, h = by+(h-w*3/4)/2, w*3/4
	}
	x0, x1 := (float64(minX)-bx)/w, (float64(maxX)-bx)/w
	y0, y1 := (float64(minY)-by)/h, (float64(maxY)-by)/h
	t.Logf("box at x %.3f..%.3f y %.3f..%.3f, %d box pixels, %d neutral, %d other", x0, x1, y0, y1, reds, neutral, other)
	for _, v := range [][2]float64{{x0, 0.25}, {x1, 0.75}, {y0, 0.25}, {y1, 0.75}} {
		if v[0] < v[1]-0.04 || v[0] > v[1]+0.04 {
			t.Errorf("box edges %.3f,%.3f,%.3f,%.3f want 0.25,0.75,0.25,0.75", x0, x1, y0, y1)
			break
		}
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
