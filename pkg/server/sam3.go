package server

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"strings"
	"time"

	goimage "github.com/aep/moxgo/pkg/image"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/aep/moxgo/pkg/sam3"

	// image decoders for request bytes
	_ "image/jpeg"
	_ "image/png"
)

// sam3PromptInput is the input carrying comma-separated text prompts.
// Models other than sam3 ignore it, so a client may always send it.
const sam3PromptInput = "prompt"

// InputError is a problem with the request's inputs rather than the server.
type InputError struct{ Err error }

func (e *InputError) Error() string { return e.Err.Error() }
func (e *InputError) Unwrap() error { return e.Err }

func inputErrorf(format string, args ...any) error {
	return &InputError{fmt.Errorf(format, args...)}
}

func validateSAM3(cfg *ModelConfig) error {
	if fi, err := os.Stat(cfg.Path); err != nil || !fi.IsDir() {
		return fmt.Errorf("pipeline sam3: path %q must be a directory with the SAM 3 onnx models", cfg.Path)
	}
	if cfg.Threshold < 0 || cfg.Threshold >= 1 {
		return fmt.Errorf("pipeline sam3: threshold must be in [0, 1)")
	}
	if cfg.Threshold == 0 {
		cfg.Threshold = 0.5
	}
	// Fixed interface, so `ls` shows what to send.
	cfg.Inputs = map[string]*InputConfig{
		"image":         {Name: "image", Type: "image"},
		sam3PromptInput: {Name: sam3PromptInput, Type: "text"},
	}
	cfg.InputList = []*InputConfig{cfg.Inputs["image"], cfg.Inputs[sam3PromptInput]}
	cfg.Outputs = map[string]*OutputConfig{"detections": {Name: "detections", Type: "detection"}}
	cfg.OutputList = []*OutputConfig{cfg.Outputs["detections"]}
	return nil
}

func (m *InstanceManager) runSAM3(cfg *ModelConfig, ep string, threads int) (*Instance, error) {
	if ep == "" {
		ep = cfg.EP
	}
	model, err := sam3.Open(m.rt, cfg.Path, sam3.Options{EP: ep, Threads: threads})
	if err != nil {
		return nil, err
	}
	fmt.Printf("  %s: sam3 ep=%s threads=%d\n", cfg.Name, model.ActiveEP, threads)
	return m.register(&Instance{
		Name:     cfg.Name,
		Config:   cfg,
		ActiveEP: model.ActiveEP,
		Threads:  threads,
		slots:    make(chan *runSlot, 1),
		sam3:     model,
	}), nil
}

// predictSAM3 takes an encoded image and optional "prompt" input (text,
// comma-separated; default: the configured prompts) and returns one
// detection per object found, labelled with the prompt that found it.
func (inst *Instance) predictSAM3(inputs []*gomaxv1.Input) (*gomaxv1.PredictResponse, error) {
	prompts := inst.Config.Prompts
	var imgIn *gomaxv1.Input
	for _, in := range inputs {
		if in.Name == sam3PromptInput {
			prompts = splitPrompts(string(in.GetFileBytes()))
		} else if imgIn == nil {
			imgIn = in
		}
	}
	if imgIn == nil {
		return nil, inputErrorf("missing input %q", "image")
	}
	if len(prompts) == 0 {
		return nil, inputErrorf("model %s: no prompts: send a %q input (e.g. \"boat,buoy\") or set prompts in the config", inst.Name, sam3PromptInput)
	}
	var img image.Image
	if yc := imgIn.GetYcbcr(); yc != nil {
		var err error
		if img, err = goimage.YCbCr420(int(yc.GetWidth()), int(yc.GetHeight()), yc.GetY(), yc.GetCb(), yc.GetCr()); err != nil {
			return nil, inputErrorf("image: %w", err)
		}
	} else {
		var err error
		if img, _, err = image.Decode(bytes.NewReader(imgIn.GetFileBytes())); err != nil {
			return nil, inputErrorf("decode image: %w", err)
		}
	}

	inst.sam3mu.Lock()
	defer inst.sam3mu.Unlock()
	inst.mu.Lock()
	closed := inst.closed
	inst.mu.Unlock()
	if closed {
		return nil, fmt.Errorf("model %s is stopped", inst.Name)
	}

	start := time.Now()
	dets, err := inst.sam3.Detect(img, prompts, inst.Config.Threshold)
	if err != nil {
		return nil, err
	}
	elapsed := time.Since(start)

	classOf := make(map[string]int32, len(prompts))
	for i, p := range prompts {
		if _, ok := classOf[p]; !ok {
			classOf[p] = int32(i)
		}
	}
	out := &gomaxv1.Detections{}
	for _, d := range dets {
		out.Items = append(out.Items, &gomaxv1.Detection{
			Label: d.Label, ClassId: classOf[d.Label], Confidence: d.Score,
			X1: d.Box[0], Y1: d.Box[1], X2: d.Box[2], Y2: d.Box[3],
		})
	}
	return &gomaxv1.PredictResponse{
		InferenceTimeMs: float64(elapsed.Microseconds()) / 1000.0,
		Outputs: []*gomaxv1.OutputResult{{
			Name:   "detections",
			Result: &gomaxv1.OutputResult_Detections{Detections: out},
		}},
	}, nil
}

func splitPrompts(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
