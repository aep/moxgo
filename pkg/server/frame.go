package server

import (
	"fmt"
	"time"

	goimage "github.com/aep/moxgo/pkg/image"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
)

// FrameTimes are the stages of one FrameRunner inference.
type FrameTimes struct {
	Inference, Postprocess time.Duration
}

// FrameRunner runs a single-image-input model on video frames: the fastest
// way into a model, taking the decoder's planes directly and reusing one run
// slot. It is for one goroutine. Close it before the model is stopped.
type FrameRunner struct {
	inst  *Instance
	slot  *runSlot
	input int
	cfg   *InputConfig
	pre   *goimage.Preprocessor
	lb    goimage.Letterbox
}

// NewFrameRunner makes a FrameRunner. It fails for models that do not take
// exactly one image, like pipelines; those go through Predict.
func (inst *Instance) NewFrameRunner() (*FrameRunner, error) {
	if inst.sam3 != nil || inst.Session == nil {
		return nil, fmt.Errorf("model %s has no single-image session", inst.Name)
	}
	var cfg *InputConfig
	input := -1
	for i, in := range inst.Config.InputList {
		if in.Type != "image" {
			return nil, fmt.Errorf("model %s has a %s input", inst.Name, in.Type)
		}
		if cfg != nil {
			return nil, fmt.Errorf("model %s has several inputs", inst.Name)
		}
		cfg, input = in, i
	}
	if cfg == nil {
		return nil, fmt.Errorf("model %s has no image input", inst.Name)
	}
	slot := inst.acquireSlot()
	if !slot.warmed {
		if err := inst.warmSlot(slot); err != nil {
			inst.releaseSlot(slot)
			return nil, err
		}
	}
	return &FrameRunner{
		inst: inst, slot: slot, input: input, cfg: cfg,
		pre: slot.preprocessor(input, cfg),
	}, nil
}

// InputSize is the model's image size.
func (r *FrameRunner) InputSize() (width, height int) { return r.cfg.Width, r.cfg.Height }

// Prepare makes p the input of the next Run. p is not used after it returns.
func (r *FrameRunner) Prepare(p *goimage.Planes) goimage.Letterbox {
	r.lb = r.pre.Process(r.slot.bufs[r.input], p)
	return r.lb
}

// Run infers on the prepared picture. Detection coordinates in the response
// are pixels of the picture's scaled size in the model input, that is
// Letterbox.ContentW×ContentH.
func (r *FrameRunner) Run() (*gomaxv1.PredictResponse, FrameTimes, error) {
	var t FrameTimes
	inst := r.inst
	start := time.Now()
	if err := r.slot.run.Exec(); err != nil {
		return nil, t, fmt.Errorf("exec: %w", err)
	}
	t.Inference = time.Since(start)
	start = time.Now()
	tensor := r.lb.ContentTensor(r.cfg.Width, r.cfg.Height)
	resp := &gomaxv1.PredictResponse{InferenceTimeMs: float64(t.Inference.Microseconds()) / 1000}
	for i, out := range inst.Session.Outputs {
		oc := inst.Config.Outputs[out.Name]
		if oc == nil {
			continue
		}
		result, err := r.slot.run.GetOutput(i)
		if err != nil {
			continue
		}
		resp.Outputs = append(resp.Outputs, buildOutputResult(out.Name, oc, result, tensor, r.slot))
	}
	t.Postprocess = time.Since(start)
	return resp, t, nil
}

// Close gives the slot back.
func (r *FrameRunner) Close() {
	if r.slot != nil {
		r.inst.releaseSlot(r.slot)
		r.slot = nil
	}
}
