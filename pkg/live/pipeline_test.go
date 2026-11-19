package live

import (
	"context"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aep/gaal"
	"github.com/aep/gaal/display"
	"github.com/aep/moxgo/pkg/camera"
)

func TestMailboxLatestWins(t *testing.T) {
	m := NewMailbox[int]()
	if old, ok := m.Put(1); ok {
		t.Fatalf("first Put replaced %d", old)
	}
	if old, ok := m.Put(2); !ok || old != 1 {
		t.Fatalf("got %d, %v", old, ok)
	}
	m.Put(3)
	if v, ok := m.Take(); !ok || v != 3 {
		t.Fatalf("Take %d, %v", v, ok)
	}
	if _, ok := m.TryTake(); ok {
		t.Fatal("TryTake on empty")
	}
	done := make(chan int)
	go func() {
		v, _ := m.Take()
		done <- v
	}()
	time.Sleep(5 * time.Millisecond)
	m.Put(7)
	if v := <-done; v != 7 {
		t.Fatalf("waiter got %d", v)
	}
	m.Put(8)
	if v, ok := m.Close(); !ok || v != 8 {
		t.Fatalf("Close returned %d, %v", v, ok)
	}
	if _, ok := m.Take(); ok {
		t.Fatal("Take after Close")
	}
	if v, ok := m.Put(9); !ok || v != 9 {
		t.Fatalf("Put after Close: %d, %v", v, ok)
	}
}

// fakeSource produces numbered frames at a fixed pace; luma encodes the number.
type fakeSource struct {
	n, total int
	every    time.Duration
	pool     gaal.FramePool
}

func (s *fakeSource) Next(time.Duration) (*gaal.VideoFrame, camera.FrameTiming, error) {
	if s.n >= s.total {
		return nil, camera.FrameTiming{}, io.EOF
	}
	s.n++
	time.Sleep(s.every)
	f := s.pool.Get(64, 48, gaal.NV12)
	f.Range = gaal.RangeFull
	for i := range f.Y {
		f.Y[i] = byte(s.n)
	}
	for i := range f.U {
		f.U[i] = 128
	}
	now := time.Now()
	return f, camera.FrameTiming{Captured: now, Received: now, Sent: now, Decoded: now}, nil
}

func (s *fakeSource) Name() string               { return "fake" }
func (s *fakeSource) Matrix() camera.ColorMatrix { return camera.MatrixBT601 }
func (s *fakeSource) Close() error               { return nil }

// slowDetector takes much longer than the frame interval.
type slowDetector struct {
	delay time.Duration
	seen  []byte // frame numbers it got
	cur   byte
	mu    sync.Mutex
}

func (d *slowDetector) Prepare(f *gaal.VideoFrame) error {
	d.cur = f.Y[0]
	return nil
}

func (d *slowDetector) Infer(context.Context) (*Result, Times, error) {
	time.Sleep(d.delay)
	d.mu.Lock()
	d.seen = append(d.seen, d.cur)
	d.mu.Unlock()
	return &Result{Items: []Item{{Label: "x", X1: 1, Y1: 1, X2: 40, Y2: 30}}, CoordW: 64, CoordH: 48}, Times{Inference: d.delay}, nil
}
func (d *slowDetector) Close() {}

type countDisplay struct {
	shown atomic.Int64
	seen  [256]atomic.Int32
}

func (c *countDisplay) Show(f *gaal.VideoFrame) error {
	c.shown.Add(1)
	// The corner is not drawn on, so it still tells the frame number.
	c.seen[f.Y[f.YStride*47+63]].Add(1)
	return nil
}
func (c *countDisplay) CloseRequested() bool { return false }
func (c *countDisplay) Stats() display.Stats { return display.Stats{} }

// A detector much slower than the camera does not slow the video, and always
// works on the newest frame.
func TestPipelineSlowDetectorDoesNotBlockVideo(t *testing.T) {
	src := &fakeSource{total: 120, every: time.Millisecond}
	det := &slowDetector{delay: 15 * time.Millisecond}
	disp := &countDisplay{}
	st := NewStats()
	p := New(Options{Source: src, Detector: det, Display: disp, Stats: st, StartupSkip: -1})
	start := time.Now()
	if err := p.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	el := time.Since(start)
	if got := disp.shown.Load(); got < 120 {
		t.Fatalf("shown %d of 120 frames", got)
	}
	for i := 1; i <= 120; i++ {
		if disp.seen[i].Load() == 0 {
			t.Errorf("frame %d not shown", i)
		}
	}
	if el > 500*time.Millisecond {
		t.Errorf("took %v: inference held the video back", el)
	}
	det.mu.Lock()
	defer det.mu.Unlock()
	if len(det.seen) == 0 || len(det.seen) > 25 {
		t.Fatalf("detector ran %d times", len(det.seen))
	}
	for i := 1; i < len(det.seen); i++ {
		if det.seen[i] <= det.seen[i-1] {
			t.Fatalf("detector went back: %v", det.seen)
		}
	}
}

// Inference must see frames as decoded: the video path draws only on copies
// while a frame is being read. Run with -race for full effect.
type checkingDetector struct {
	bad atomic.Int32
}

func (d *checkingDetector) Prepare(f *gaal.VideoFrame) error {
	for i := 0; i < len(f.Y); i += 7 {
		if f.Y[i] != f.Y[0] {
			d.bad.Add(1)
			break
		}
	}
	return nil
}

func (d *checkingDetector) Infer(context.Context) (*Result, Times, error) {
	return &Result{Items: []Item{{Label: "x", Confidence: 0.5, X1: 2, Y1: 2, X2: 60, Y2: 40}}, CoordW: 64, CoordH: 48}, Times{}, nil
}
func (d *checkingDetector) Close() {}

func TestPipelineInferenceSeesCleanPixels(t *testing.T) {
	for _, noRedraw := range []bool{false, true} {
		src := &fakeSource{total: 300}
		det := &checkingDetector{}
		p := New(Options{Source: src, Detector: det, Display: &countDisplay{}, StartupSkip: -1, NoRedraw: noRedraw})
		if err := p.Run(context.Background()); err != nil {
			t.Fatal(err)
		}
		if n := det.bad.Load(); n != 0 {
			t.Fatalf("noRedraw=%v: %d frames were drawn on before inference read them", noRedraw, n)
		}
	}
}

// The headless display shows the drawn picture: boxes appear in the buffer
// handed to it, at the mapped position.
func TestPipelineHeadlessReadback(t *testing.T) {
	d, err := display.Open(display.Options{Backend: "headless", Width: 64, Height: 48, Matrix: display.BT601})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	src := &fakeSource{total: 40, every: 2 * time.Millisecond}
	p := New(Options{Source: src, Detector: &checkingDetector{}, Display: d, StartupSkip: -1})
	if err := p.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	img, _, err := d.Readback()
	if err != nil {
		t.Fatal(err)
	}
	// Row 20 crosses the box's left edge at x=2..3: strongly colored.
	r, g, _, _ := img.At(2, 20).RGBA()
	if r>>8 < 150 || g>>8 > 120 {
		t.Fatalf("box edge is rgb(%d,%d)", r>>8, g>>8)
	}
	r2, g2, b2, _ := img.At(30, 20).RGBA()
	if diff := int(r2>>8) - int(g2>>8); diff > 8 || diff < -8 || int(g2>>8)-int(b2>>8) > 8 {
		t.Fatalf("inside the box is tinted: %d,%d,%d", r2>>8, g2>>8, b2>>8)
	}
}
