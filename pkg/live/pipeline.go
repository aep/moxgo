package live

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/aep/gaal"
	"github.com/aep/gaal/display"
	"github.com/aep/moxgo/pkg/camera"
)

// Presenter is where video goes. *display.Display is one.
type Presenter interface {
	Show(*gaal.VideoFrame) error
	CloseRequested() bool
	Stats() display.Stats
}

// Options configures a Pipeline.
type Options struct {
	Source   camera.VideoSource
	Detector Detector  // nil shows video only
	Display  Presenter // nil runs without showing
	Stats    *Stats    // nil: none

	// MaxDetectFPS limits how often inference starts; 0 is no limit.
	MaxDetectFPS float64
	// MaxResultAge drops results older than this from the picture. Default 2 s.
	MaxResultAge time.Duration
	// Duration ends the run; 0 runs until ctx is done.
	Duration time.Duration
	// StartupSkip is excluded from the statistics. Default 1 s.
	StartupSkip time.Duration
	// StatsEvery prints a line of statistics to StatsOut; 0 prints none.
	StatsEvery time.Duration
	StatsOut   io.Writer
	// NoRedraw disables showing a frame again when its own detections arrive
	// before the next frame. Without that, boxes are one frame behind.
	NoRedraw bool
	// Log receives occasional non-fatal problems, already redacted.
	Log func(string)
}

// liveFrame is a frame travelling to inference. refs counts the video path
// and the inference path. reading is nonzero until inference has finished
// reading the pixels, or dropped the frame.
type liveFrame struct {
	f         *gaal.VideoFrame
	t         camera.FrameTiming
	seq       uint64
	published time.Time
	refs      atomic.Int32
	reading   atomic.Int32
}

// published is the newest detection output.
type publishedResult struct {
	res      *Result
	seq      uint64
	captured time.Time
}

// Pipeline moves video from a source to the display and frames to a detector.
type Pipeline struct {
	o      Options
	st     *Stats
	mbox   *Mailbox[*liveFrame]
	latest atomic.Pointer[publishedResult]

	freeMu sync.Mutex
	free   []*liveFrame

	// showMu serializes composing and showing, by the video loop and by the
	// redraw after an inference. It guards the fields below.
	showMu   sync.Mutex
	copies   gaal.FramePool
	overlay  Overlay
	current  *gaal.VideoFrame // newest frame, undrawn, retained
	curSeq   uint64
	curBT709 bool
	shownRes *publishedResult // result drawn into the last Show
	age      time.Duration    // age of the detections last shown
	haveAge  bool
}

// New makes a pipeline. It does not start anything.
func New(o Options) *Pipeline {
	if o.MaxResultAge <= 0 {
		o.MaxResultAge = 2 * time.Second
	}
	if o.StartupSkip == 0 {
		o.StartupSkip = time.Second
	}
	st := o.Stats
	if st == nil {
		st = NewStats()
	}
	return &Pipeline{o: o, st: st, mbox: NewMailbox[*liveFrame]()}
}

func (p *Pipeline) getFrame() *liveFrame {
	p.freeMu.Lock()
	defer p.freeMu.Unlock()
	if n := len(p.free); n > 0 {
		lf := p.free[n-1]
		p.free = p.free[:n-1]
		return lf
	}
	return &liveFrame{}
}

// unref drops one reference to lf; the last one recycles it.
func (p *Pipeline) unref(lf *liveFrame) {
	if lf.refs.Add(-1) != 0 {
		return
	}
	lf.f.Release()
	lf.f = nil
	p.freeMu.Lock()
	p.free = append(p.free, lf)
	p.freeMu.Unlock()
}

// dropReader releases the inference path's claim on lf.
func (p *Pipeline) dropReader(lf *liveFrame) {
	lf.reading.Store(0)
	p.unref(lf)
}

// Run runs until ctx is done, the duration passed, the display window closes,
// or the source fails. A source running out of frames is not an error.
func (p *Pipeline) Run(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	if p.o.Detector != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.detectLoop(ctx)
		}()
	}
	if p.o.StatsEvery > 0 && p.o.StatsOut != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			t := time.NewTicker(p.o.StatsEvery)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					p.st.PrintInterval(p.o.StatsOut)
				}
			}
		}()
	}

	err := p.videoLoop(ctx)

	cancel()
	if lf, ok := p.mbox.Close(); ok {
		p.dropReader(lf)
	}
	wg.Wait()
	return err
}

func (p *Pipeline) videoLoop(ctx context.Context) error {
	src := p.o.Source
	var seq uint64
	var presented uint64
	var started time.Time
	statsReset := false
	var deadline time.Time
	var lastErr time.Time
	defer func() {
		p.showMu.Lock()
		if p.current != nil {
			p.current.Release()
			p.current = nil
		}
		p.showMu.Unlock()
	}()
	for {
		if ctx.Err() != nil {
			return nil
		}
		if !deadline.IsZero() && time.Now().After(deadline) {
			return nil
		}
		if p.o.Display != nil && p.o.Display.CloseRequested() {
			return nil
		}
		f, t, err := src.Next(500 * time.Millisecond)
		if err != nil {
			if errors.Is(err, camera.ErrTimeout) {
				if time.Since(lastErr) > 5*time.Second {
					lastErr = time.Now()
					p.logf("no video: waiting for the source")
				}
				continue
			}
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		if started.IsZero() {
			started = time.Now()
			if p.o.Duration > 0 {
				deadline = started.Add(p.o.StartupSkip + p.o.Duration)
			}
		}
		if !statsReset && time.Since(started) >= p.o.StartupSkip {
			statsReset = true
			p.st.Reset()
			if ds, ok := p.o.Display.(interface{ ResetStats() }); ok {
				ds.ResetStats()
			}
			presented = 0
		}
		seq++

		st := p.st
		st.Frame()
		if !t.Sent.IsZero() && !t.Captured.IsZero() {
			st.Add(StageReceive, t.Sent.Sub(t.Captured))
		}
		if !t.Sent.IsZero() && !t.Decoded.IsZero() {
			st.Add(StageDecode, t.Decoded.Sub(t.Sent))
		}

		// Inference gets the frame first, sharing its pixels.
		var lf *liveFrame
		if p.o.Detector != nil {
			lf = p.getFrame()
			f.Retain()
			lf.f, lf.t, lf.seq, lf.published = f, t, seq, time.Now()
			lf.reading.Store(1)
			lf.refs.Store(2) // video path, mailbox/inference path
			if old, replaced := p.mbox.Put(lf); replaced {
				p.dropReader(old)
			}
		}

		p.showMu.Lock()
		if p.haveAge {
			st.Add(StageDetAge, p.age)
		}
		if p.current != nil {
			p.current.Release()
		}
		f.Retain()
		p.current, p.curSeq = f, seq
		p.curBT709 = matrixIsBT709(src.Matrix(), f.Width, f.Height)
		res := p.usableResult()
		showErr := p.compose(res, lf, t)
		p.showMu.Unlock()
		if lf != nil {
			p.unref(lf)
		}
		f.Release()
		if showErr != nil {
			if errors.Is(showErr, display.ErrClosed) {
				return nil
			}
			return fmt.Errorf("display: %w", showErr)
		}
		if d := p.o.Display; d != nil {
			ds := d.Stats()
			if ds.Presented > presented {
				presented = ds.Presented
				if ds.LatencyLast > 0 {
					st.Add(StagePresent, ds.LatencyLast)
					st.Add(StageDisplayLatency, time.Since(t.Captured)+ds.LatencyLast)
				}
			}
		}
	}
}

// usableResult is the newest result, unless too old to mean anything.
func (p *Pipeline) usableResult() *publishedResult {
	r := p.latest.Load()
	if r == nil || time.Since(r.captured) >= p.o.MaxResultAge {
		return nil
	}
	return r
}

// compose draws res over a copy of the current frame, or the frame itself when
// that is safe, and shows it. The caller holds showMu. lf is the frame's trip
// to inference, nil on a redraw and without a detector.
func (p *Pipeline) compose(res *publishedResult, lf *liveFrame, t camera.FrameTiming) error {
	st := p.st
	f := p.current
	show := f
	var copied *gaal.VideoFrame
	if res != nil && len(res.res.Items) > 0 {
		target := f
		// The frame must stay as decoded while inference may read it or a
		// redraw may need it.
		if p.o.Detector != nil && (!p.o.NoRedraw || lf == nil || lf.reading.Load() != 0) {
			t0 := time.Now()
			copied = p.copies.Get(f.Width, f.Height, f.Format)
			if err := f.CopyTo(copied); err != nil {
				copied.Release()
				copied = nil
			} else {
				target = copied
				st.Copied()
			}
			st.Add(StageCopy, time.Since(t0))
		}
		t0 := time.Now()
		p.overlay.Draw(target, res.res.Items, res.res.CoordW, res.res.CoordH, p.curBT709)
		st.Add(StageDraw, time.Since(t0))
		st.Drawn()
		show = target
	}
	if res != nil {
		p.age, p.haveAge = time.Since(res.captured), true
	}
	p.shownRes = res
	defer func() {
		if copied != nil {
			copied.Release()
		}
	}()
	if d := p.o.Display; d != nil {
		t0 := time.Now()
		err := d.Show(show)
		now := time.Now()
		if err != nil {
			return err
		}
		st.Add(StageShow, now.Sub(t0))
		if lf != nil || p.o.Detector == nil {
			st.Shown()
			st.Add(StageVideoLatency, now.Sub(t.Captured))
		}
		return nil
	}
	if lf != nil || p.o.Detector == nil {
		st.Add(StageVideoLatency, time.Since(t.Captured))
	}
	return nil
}

// redraw shows the current frame again if res belongs to it, so its boxes are
// its own instead of the previous frame's. Video is not held up: this runs on
// the detection goroutine and only ever replaces the picture just shown.
func (p *Pipeline) redraw(res *publishedResult) {
	p.showMu.Lock()
	defer p.showMu.Unlock()
	if p.current == nil || res.seq != p.curSeq {
		return
	}
	if len(res.res.Items) == 0 && (p.shownRes == nil || len(p.shownRes.res.Items) == 0) {
		p.age, p.haveAge = time.Since(res.captured), true
		p.shownRes = res
		return // the picture would not change
	}
	if err := p.compose(res, nil, camera.FrameTiming{}); err != nil {
		p.logf("display: %v", err)
	}
}

func (p *Pipeline) detectLoop(ctx context.Context) {
	d := p.o.Detector
	st := p.st
	var last time.Time
	var interval time.Duration
	if p.o.MaxDetectFPS > 0 {
		interval = time.Duration(float64(time.Second) / p.o.MaxDetectFPS)
	}
	var lastLog time.Time
	for {
		lf, ok := p.mbox.Take()
		if !ok {
			return
		}
		if interval > 0 {
			if w := interval - time.Since(last); w > 0 {
				// Not due yet: this frame is dropped, a newer one will be there.
				p.dropReader(lf)
				select {
				case <-ctx.Done():
					return
				case <-time.After(w):
				}
				continue
			}
		}
		last = time.Now()
		st.Add(StageQueue, last.Sub(lf.published))
		t0 := time.Now()
		err := d.Prepare(lf.f)
		pre := time.Since(t0)
		seq, captured := lf.seq, lf.t.Captured
		p.dropReader(lf)
		if err != nil {
			st.DetectError()
			continue
		}
		st.Add(StagePreprocess, pre)
		res, times, err := d.Infer(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			st.DetectError()
			if time.Since(lastLog) > 5*time.Second {
				lastLog = time.Now()
				p.logf("detection failed: %v", err)
			}
			continue
		}
		st.Add(StageInference, times.Inference)
		if times.Postprocess > 0 {
			st.Add(StagePostprocess, times.Postprocess)
		}
		st.Detection(len(res.Items))
		st.Add(StageDetLatency, time.Since(captured))
		pub := &publishedResult{res: res, seq: seq, captured: captured}
		p.latest.Store(pub)
		if !p.o.NoRedraw {
			p.redraw(pub)
		}
	}
}

func (p *Pipeline) logf(format string, args ...any) {
	if p.o.Log != nil {
		p.o.Log(camera.Redact(fmt.Sprintf(format, args...)))
	}
}
