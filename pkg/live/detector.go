package live

import (
	"context"
	"time"

	"github.com/aep/gaal"
	"github.com/aep/moxgo/pkg/camera"
	goimage "github.com/aep/moxgo/pkg/image"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/aep/moxgo/pkg/server"
)

// Result is what one inference found, ready to draw.
type Result struct {
	Items          []Item
	CoordW, CoordH float64 // the coordinate space of Items
}

// Times are the stages of one detection.
type Times struct {
	Preprocess, Inference, Postprocess time.Duration
}

// Detector turns frames into results, one at a time.
type Detector interface {
	// Prepare takes what it needs from f, which it must not touch after
	// returning: the video path draws into frames.
	Prepare(f *gaal.VideoFrame) error
	// Infer works on the last prepared frame.
	Infer(ctx context.Context) (*Result, Times, error)
	Close()
}

// planesOf views a frame for preprocessing.
func planesOf(f *gaal.VideoFrame, bt709 bool) goimage.Planes {
	p := goimage.Planes{
		Width: f.Width, Height: f.Height,
		Y: f.Y, YStride: f.YStride,
		U: f.U, UVStride: f.UVStride,
		Limited: f.Range != gaal.RangeFull,
		BT709:   bt709,
	}
	if f.Format != gaal.NV12 {
		p.V = f.V
	}
	return p
}

// matrixIsBT709 resolves a source's matrix for a frame size, the way the
// display resolves its automatic matrix.
func matrixIsBT709(m camera.ColorMatrix, w, h int) bool {
	switch m {
	case camera.MatrixBT601:
		return false
	case camera.MatrixBT709:
		return true
	}
	return w > 1024 || h > 576
}

// resultFrom converts a response. Classifications become labels without a box.
func resultFrom(resp *gomaxv1.PredictResponse, coordW, coordH float64) *Result {
	r := &Result{CoordW: coordW, CoordH: coordH}
	for _, out := range resp.GetOutputs() {
		for _, d := range out.GetDetections().GetItems() {
			r.Items = append(r.Items, Item{
				Label: d.GetLabel(), Class: d.GetClassId(), Confidence: d.GetConfidence(),
				X1: d.GetX1(), Y1: d.GetY1(), X2: d.GetX2(), Y2: d.GetY2(),
				Mask: d.GetMask(),
			})
		}
		items := out.GetClassifications().GetItems()
		for i, c := range items {
			if i == 3 {
				break
			}
			r.Items = append(r.Items, Item{Label: c.GetLabel(), Class: c.GetClassId(), Confidence: c.GetScore()})
		}
	}
	return r
}

// --- in-process ---

// LocalDetector runs a model in this process, reading frames in place.
type LocalDetector struct {
	r      *server.FrameRunner
	matrix camera.ColorMatrix
	lb     goimage.Letterbox
	planes goimage.Planes
}

// NewLocalDetector uses inst, or fails if it needs the generic path.
func NewLocalDetector(inst *server.Instance, m camera.ColorMatrix) (*LocalDetector, error) {
	r, err := inst.NewFrameRunner()
	if err != nil {
		return nil, err
	}
	return &LocalDetector{r: r, matrix: m}, nil
}

// Prepare implements Detector.
func (d *LocalDetector) Prepare(f *gaal.VideoFrame) error {
	d.planes = planesOf(f, matrixIsBT709(d.matrix, f.Width, f.Height))
	d.lb = d.r.Prepare(&d.planes)
	d.planes = goimage.Planes{}
	return nil
}

// Infer implements Detector.
func (d *LocalDetector) Infer(context.Context) (*Result, Times, error) {
	resp, t, err := d.r.Run()
	if err != nil {
		return nil, Times{}, err
	}
	return resultFrom(resp, float64(d.lb.ContentW), float64(d.lb.ContentH)),
		Times{Inference: t.Inference, Postprocess: t.Postprocess}, nil
}

// Close implements Detector.
func (d *LocalDetector) Close() { d.r.Close() }

// --- through Predict ---

// PredictFunc runs one request, against a server or in-process.
type PredictFunc func(ctx context.Context, req *gomaxv1.PredictRequest) (*gomaxv1.PredictResponse, error)

// RemoteDetector sends frames as YCbCrImage inputs, reduced by an integer
// factor to what the model needs, and works with any model Predict can run.
type RemoteDetector struct {
	predict PredictFunc
	model   string
	input   string
	prompt  string
	// MaxW and MaxH, if set, are the model's input size: a frame at least
	// twice as large is reduced before it is sent.
	MaxW, MaxH int

	y, u, v []byte
	w, h    int
}

// NewRemoteDetector makes a detector over predict.
func NewRemoteDetector(predict PredictFunc, model, input, prompt string) *RemoteDetector {
	return &RemoteDetector{predict: predict, model: model, input: input, prompt: prompt}
}

func grow(b []byte, n int) []byte {
	if cap(b) < n {
		return make([]byte, n)
	}
	return b[:n]
}

// Prepare implements Detector: it extracts I420, full range as the wire format
// says, at a reduced size.
func (d *RemoteDetector) Prepare(f *gaal.VideoFrame) error {
	k := 1
	if d.MaxW > 0 && d.MaxH > 0 {
		k = max(1, min(f.Width/d.MaxW, f.Height/d.MaxH))
	}
	w, h := f.Width/k, f.Height/k
	cw, ch := (w+1)/2, (h+1)/2
	d.w, d.h = w, h
	d.y, d.u, d.v = grow(d.y, w*h), grow(d.u, cw*ch), grow(d.v, cw*ch)
	lim := f.Range != gaal.RangeFull
	for y := range h {
		s := f.Y[(y*k)*f.YStride:]
		o := d.y[y*w : (y+1)*w]
		for x := range o {
			v := s[x*k]
			if lim {
				v = fullY[v]
			}
			o[x] = v
		}
	}
	step, off := 1, 0
	if f.Format == gaal.NV12 {
		step, off = 2, 1
	}
	vp := f.V
	if f.Format == gaal.NV12 {
		vp = f.U
	}
	for y := range ch {
		us := f.U[y*k*f.UVStride:]
		vs := vp[y*k*f.UVStride+off:]
		uo, vo := d.u[y*cw:(y+1)*cw], d.v[y*cw:(y+1)*cw]
		for x := range uo {
			a, b := us[x*k*step], vs[x*k*step]
			if lim {
				a, b = fullC[a], fullC[b]
			}
			uo[x], vo[x] = a, b
		}
	}
	return nil
}

// Request is the request for the last prepared frame. Its planes are reused
// by the next Prepare.
func (d *RemoteDetector) Request() *gomaxv1.PredictRequest {
	in := []*gomaxv1.Input{{
		Name: d.input,
		Data: &gomaxv1.Input_Ycbcr{Ycbcr: &gomaxv1.YCbCrImage{
			Width: uint32(d.w), Height: uint32(d.h), Y: d.y, Cb: d.u, Cr: d.v,
		}},
	}}
	if d.prompt != "" {
		if in[0].Name == "" {
			in[0].Name = "image"
		}
		in = append(in, &gomaxv1.Input{Name: "prompt", Data: &gomaxv1.Input_FileBytes{FileBytes: []byte(d.prompt)}})
	}
	return &gomaxv1.PredictRequest{Model: d.model, Inputs: in}
}

// Infer implements Detector.
func (d *RemoteDetector) Infer(ctx context.Context) (*Result, Times, error) {
	start := time.Now()
	resp, err := d.predict(ctx, d.Request())
	if err != nil {
		return nil, Times{}, err
	}
	return resultFrom(resp, float64(d.w), float64(d.h)), Times{Inference: time.Since(start)}, nil
}

// Close implements Detector.
func (d *RemoteDetector) Close() {}

// tables expanding limited range to full.
var fullY, fullC = func() (y, c [256]byte) {
	for i := range 256 {
		y[i] = to8((float64(i) - 16) * 255 / 219)
		c[i] = to8((float64(i)-128)*255/224 + 128)
	}
	return y, c
}()
