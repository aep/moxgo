package server

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/aep/moxgo/pkg/audio"
	goimage "github.com/aep/moxgo/pkg/image"
	"github.com/aep/moxgo/pkg/onnx"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/aep/moxgo/pkg/sam3"
	"go.opentelemetry.io/otel/codes"
)

const poolSize = 4

type runSlot struct {
	run    *onnx.Run
	bufs   [][]float32
	shapes [][]int64
	warmed bool
}

// Instance is a running model (keyed by model name).
type Instance struct {
	Name     string
	Config   *ModelConfig
	Session  *onnx.Session
	ActiveEP string
	Threads  int

	slots  chan *runSlot
	mu     sync.Mutex
	closed bool

	// sam3 is set instead of Session for a sam3 pipeline.
	sam3   *sam3.Model
	sam3mu sync.Mutex // serializes sam3 runs: each one needs GBs of VRAM
}

// defaultThreads is the intra-op threads of models that set none: two
// roughly halves inference time on phones' CPUs over one, and leaves cores
// for the rest of the device.
const defaultThreads = 2

// InstanceManager manages running model instances.
type InstanceManager struct {
	rt       *onnx.Runtime
	registry *Registry

	mu        sync.RWMutex
	instances map[string]*Instance

	// epResolver, if set, picks the execution providers of models that
	// name none (see Server.SetEPResolver).
	epResolver EPResolver
}

// EPResolver returns the execution providers to try for a model, in order;
// the CPU is the fallback. ok false leaves the choice to moxgo, which tries
// all providers the runtime has.
type EPResolver func(cfg *ModelConfig) (eps []onnx.ExecutionProvider, ok bool, err error)

func NewInstanceManager(rt *onnx.Runtime, reg *Registry) *InstanceManager {
	return &InstanceManager{
		rt:        rt,
		registry:  reg,
		instances: make(map[string]*Instance),
	}
}

// Run loads a model by name. Returns existing instance if already running.
// The ep parameter overrides the execution provider for initial load; it is
// ignored if the model is already running.
func (m *InstanceManager) Run(modelName string, ep string) (*Instance, error) {
	modelName = ModelName(modelName)
	m.mu.RLock()
	if inst, ok := m.instances[modelName]; ok {
		m.mu.RUnlock()
		return inst, nil
	}
	m.mu.RUnlock()

	cfg, ok := m.registry.Get(modelName)
	if !ok {
		return nil, fmt.Errorf("model %q not found in registry", modelName)
	}

	threads := cfg.Threads
	if threads <= 0 {
		threads = defaultThreads
	}

	if cfg.Pipeline == "sam3" {
		return m.runSAM3(cfg, ep, threads)
	}

	var eps []onnx.ExecutionProvider
	resolved := false
	if ep == "" && cfg.EP == "" && m.epResolver != nil {
		var err error
		eps, resolved, err = m.epResolver(cfg)
		if err != nil {
			fmt.Printf("  %s: runtime: %v; using the providers the runtime has\n", modelName, err)
			eps, resolved = nil, false
		}
	}
	if resolved {
		// eps as the runtime rules give them
	} else if ep != "" {
		eps = append(eps, onnx.ExecutionProvider{Name: ep})
	} else if cfg.EP != "" {
		eps = append(eps, onnx.ExecutionProvider{Name: cfg.EP})
	} else {
		// Auto-try available GPU providers
		if providers, err := m.rt.GetAvailableProviders(); err == nil {
			for _, p := range providers {
				if p != "CPUExecutionProvider" {
					eps = append(eps, onnx.ExecutionProvider{
						Name: strings.TrimSuffix(p, "ExecutionProvider"),
					})
				}
			}
		}
	}

	sess, err := m.rt.OpenSessionWith(cfg.Path, onnx.SessionOptions{
		ExecutionProviders: eps,
		IntraOpThreads:     threads,
		LogLevel:           onnx.OrtLoggingLevelError,
	})
	if err != nil {
		return nil, fmt.Errorf("open session: %w (%s)", err, m.rt.LastError())
	}
	for _, e := range sess.EPErrors {
		fmt.Printf("  %s: ep %s failed: %v\n", modelName, e.Name, e.Err)
	}
	fmt.Printf("  %s: ep=%s threads=%d\n", modelName, sess.ActiveEP, threads)

	inst := &Instance{
		Name:     modelName,
		Config:   cfg,
		Session:  sess,
		ActiveEP: sess.ActiveEP,
		Threads:  threads,
		slots:    make(chan *runSlot, poolSize),
	}

	// A provider can take a model and still fail to run it (QNN on a
	// model with dynamic shapes, say): it runs on the CPU then.
	if err := inst.tryRun(); err != nil && sess.ActiveEP != "CPU" {
		fmt.Printf("  %s: does not run on %s (%v); on the CPU instead\n", modelName, sess.ActiveEP, err)
		sess.Close()
		cpu, cerr := m.rt.OpenSessionWith(cfg.Path, onnx.SessionOptions{
			IntraOpThreads: threads,
			LogLevel:       onnx.OrtLoggingLevelError,
		})
		if cerr != nil {
			return nil, fmt.Errorf("open session on the CPU: %w (%s)", cerr, m.rt.LastError())
		}
		inst.Session, inst.ActiveEP = cpu, cpu.ActiveEP
		fmt.Printf("  %s: ep=%s threads=%d\n", modelName, cpu.ActiveEP, threads)
	}

	return m.register(inst), nil
}

// tryRun runs the model once on zeros, as the warmup of its first slot,
// which it keeps. A model with an input of no fixed size is not tried.
func (inst *Instance) tryRun() error {
	slot := inst.newSlot()
	for _, b := range slot.bufs {
		if b == nil {
			slot.run.Close()
			return nil
		}
	}
	if err := inst.warmSlot(slot); err != nil {
		slot.run.Close()
		return err
	}
	inst.releaseSlot(slot)
	return nil
}

// register adds a new instance, or returns (and keeps) one that another
// caller loaded meanwhile, closing inst.
func (m *InstanceManager) register(inst *Instance) *Instance {
	m.mu.Lock()
	if existing, ok := m.instances[inst.Name]; ok {
		m.mu.Unlock()
		inst.close()
		return existing
	}
	m.instances[inst.Name] = inst
	m.mu.Unlock()
	return inst
}

// GetOrRun returns a running instance, auto-loading if needed.
func (m *InstanceManager) GetOrRun(modelName string) (*Instance, error) {
	modelName = ModelName(modelName)
	m.mu.RLock()
	inst, ok := m.instances[modelName]
	m.mu.RUnlock()
	if ok {
		return inst, nil
	}
	return m.Run(modelName, "")
}

// Rm stops and removes a model instance.
func (m *InstanceManager) Rm(modelName string) error {
	modelName = ModelName(modelName)
	m.mu.Lock()
	inst, ok := m.instances[modelName]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("model %q not running", modelName)
	}
	delete(m.instances, modelName)
	m.mu.Unlock()

	inst.close()
	return nil
}

// List returns all running instances.
func (m *InstanceManager) List() []*Instance {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]*Instance, 0, len(m.instances))
	for _, inst := range m.instances {
		out = append(out, inst)
	}
	return out
}

// Shutdown closes all instances.
func (m *InstanceManager) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, inst := range m.instances {
		inst.close()
	}
	m.instances = make(map[string]*Instance)
}

func (inst *Instance) acquireSlot() *runSlot {
	select {
	case s := <-inst.slots:
		return s
	default:
		return inst.newSlot()
	}
}

func (inst *Instance) releaseSlot(s *runSlot) {
	select {
	case inst.slots <- s:
	default:
		// Pool full — close this slot to release the Pinner
		s.run.Close()
	}
}

func (inst *Instance) newSlot() *runSlot {
	slot := &runSlot{
		run:    inst.Session.NewRun(),
		bufs:   make([][]float32, len(inst.Config.InputList)),
		shapes: make([][]int64, len(inst.Config.InputList)),
	}
	for i, inp := range inst.Config.InputList {
		size := inp.InputSize()
		if size > 0 {
			slot.bufs[i] = make([]float32, size)
			slot.shapes[i] = inp.InputShape()
		}
	}
	return slot
}

func (inst *Instance) warmSlot(slot *runSlot) error {
	for i, inp := range inst.Config.InputList {
		if slot.bufs[i] == nil {
			continue
		}
		if err := onnx.BindSlice(slot.run, inp.Name, slot.bufs[i], slot.shapes[i]); err != nil {
			return fmt.Errorf("bind input %q: %w", inp.Name, err)
		}
	}
	if err := slot.run.Exec(); err != nil {
		return fmt.Errorf("warmup exec: %w", err)
	}
	slot.warmed = true
	return nil
}

// Predict runs single-shot inference.
func (inst *Instance) Predict(inputs []*gomaxv1.Input) (*gomaxv1.PredictResponse, error) {
	return inst.PredictContext(context.Background(), inputs)
}

// PredictContext is Predict, traced under ctx's span (see trace.go).
func (inst *Instance) PredictContext(ctx context.Context, inputs []*gomaxv1.Input) (resp *gomaxv1.PredictResponse, err error) {
	ctx, span := tracer.Start(ctx, "moxgo.predict", inst.spanAttrs())
	defer func() {
		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
		}
		span.End()
	}()

	inst.mu.Lock()
	if inst.closed {
		inst.mu.Unlock()
		return nil, fmt.Errorf("model %s is stopped", inst.Name)
	}
	inst.mu.Unlock()

	if inst.sam3 != nil {
		return inst.predictSAM3(inputs)
	}
	inputs = withoutPrompt(inst.Config, inputs)

	_, wait := tracer.Start(ctx, "moxgo.wait_slot")
	slot := inst.acquireSlot()
	wait.End()
	defer inst.releaseSlot(slot)

	_, pre := tracer.Start(ctx, "moxgo.preprocess")
	inputBytes := 0
	for _, in := range inputs {
		yc := in.GetYcbcr()
		inputBytes += len(in.GetFileBytes()) + len(yc.GetY()) + len(yc.GetCb()) + len(yc.GetCr())
	}
	pre.SetAttributes(attrInputSize.Int(inputBytes))
	var imgTensors []*goimage.Tensor
	for i, inp := range inst.Config.InputList {
		reqInput := findInput(inputs, inp.Name)
		if reqInput == nil {
			return nil, inputErrorf("missing input %q", inp.Name)
		}

		switch inp.Type {
		case "image":
			var tensor *goimage.Tensor
			if yc := reqInput.GetYcbcr(); yc != nil {
				img, err := goimage.YCbCr420(int(yc.GetWidth()), int(yc.GetHeight()), yc.GetY(), yc.GetCb(), yc.GetCr())
				if err != nil {
					pre.End()
					return nil, inputErrorf("image %q: %w", inp.Name, err)
				}
				tensor = goimage.FromImage(img, inp.Width, inp.Height)
			} else {
				var err error
				tensor, err = goimage.Decode(bytes.NewReader(reqInput.GetFileBytes()), inp.Width, inp.Height)
				if err != nil {
					pre.End()
					return nil, inputErrorf("decode image %q: %w", inp.Name, err)
				}
			}
			copy(slot.bufs[i], tensor.Data)
			if len(inp.Mean) > 0 && len(inp.Std) > 0 {
				NormalizeCHW(slot.bufs[i], inp.Width, inp.Height, inp.Mean, inp.Std)
			}
			imgTensors = append(imgTensors, tensor)

		case "audio":
			at, err := audio.Decode(bytes.NewReader(reqInput.GetFileBytes()), audio.Options{
				SampleRate: inp.SampleRate,
				MaxSamples: inp.InputSize(),
			})
			if err != nil {
				return nil, inputErrorf("decode audio %q: %w", inp.Name, err)
			}
			n := copy(slot.bufs[i], at.Data)
			for j := n; j < len(slot.bufs[i]); j++ {
				slot.bufs[i][j] = 0
			}

		case "raw":
			rawData := rawTensorToFloat32(reqInput.GetTensor())
			if len(rawData) != len(slot.bufs[i]) {
				pre.End()
				return nil, inputErrorf("input %q: size %d != expected %d", inp.Name, len(rawData), len(slot.bufs[i]))
			}
			copy(slot.bufs[i], rawData)
		}
	}
	pre.End()

	if !slot.warmed {
		_, warm := tracer.Start(ctx, "moxgo.warmup")
		err := inst.warmSlot(slot)
		warm.End()
		if err != nil {
			return nil, err
		}
	}

	_, infer := tracer.Start(ctx, "moxgo.inference")
	start := time.Now()
	err = slot.run.Exec()
	elapsed := time.Since(start)
	infer.End()
	if err != nil {
		return nil, fmt.Errorf("exec: %w", err)
	}

	var firstImg *goimage.Tensor
	if len(imgTensors) > 0 {
		firstImg = imgTensors[0]
	}

	_, post := tracer.Start(ctx, "moxgo.postprocess")
	defer post.End()
	resp = &gomaxv1.PredictResponse{
		InferenceTimeMs: float64(elapsed.Microseconds()) / 1000.0,
	}
	defer func() { post.SetAttributes(attrOutputs.Int(len(resp.GetOutputs()))) }()
	for i, out := range inst.Session.Outputs {
		oc := inst.Config.Outputs[out.Name]
		if oc == nil {
			continue
		}
		result, err := slot.run.GetOutput(i)
		if err != nil {
			continue
		}
		resp.Outputs = append(resp.Outputs, buildOutputResult(out.Name, oc, result, firstImg, slot))
	}
	return resp, nil
}

// PredictChunked runs chunked audio inference.
func (inst *Instance) PredictChunked(inputs []*gomaxv1.Input, fn func(*gomaxv1.PredictStreamResponse) error) error {
	inst.mu.Lock()
	if inst.closed {
		inst.mu.Unlock()
		return fmt.Errorf("model %s is stopped", inst.Name)
	}
	inst.mu.Unlock()

	if inst.sam3 != nil || len(inst.Config.InputList) != 1 || inst.Config.InputList[0].Type != "audio" {
		return fmt.Errorf("streaming only supported for single-audio-input models")
	}
	inp := inst.Config.InputList[0]
	reqInput := findInput(inputs, inp.Name)
	if reqInput == nil {
		return fmt.Errorf("missing input %q", inp.Name)
	}

	at, err := audio.Decode(bytes.NewReader(reqInput.GetFileBytes()), audio.Options{
		SampleRate: inp.SampleRate,
	})
	if err != nil {
		return fmt.Errorf("decode audio: %w", err)
	}

	chunkSamples := inp.InputSize()
	overlapSamples := int(inp.Overlap * float64(inp.SampleRate))
	stride := chunkSamples - overlapSamples
	if stride <= 0 {
		stride = chunkSamples / 2
	}

	slot := inst.acquireSlot()
	defer inst.releaseSlot(slot)

	if !slot.warmed {
		if err := inst.warmSlot(slot); err != nil {
			return err
		}
	}

	for start := 0; start < len(at.Data); start += stride {
		end := start + chunkSamples
		if end > len(at.Data) {
			n := copy(slot.bufs[0], at.Data[start:])
			for j := n; j < len(slot.bufs[0]); j++ {
				slot.bufs[0][j] = 0
			}
		} else {
			copy(slot.bufs[0], at.Data[start:end])
		}

		if err := slot.run.Exec(); err != nil {
			return fmt.Errorf("exec at %.1fs: %w", float64(start)/float64(at.SampleRate), err)
		}

		tStart := float64(start) / float64(at.SampleRate)
		tEnd := float64(end) / float64(at.SampleRate)
		if tEnd > at.Duration {
			tEnd = at.Duration
		}

		resp := &gomaxv1.PredictStreamResponse{
			WindowStartSec: tStart,
			WindowEndSec:   tEnd,
		}
		for i, out := range inst.Session.Outputs {
			oc := inst.Config.Outputs[out.Name]
			if oc == nil {
				continue
			}
			result, err := slot.run.GetOutput(i)
			if err != nil {
				continue
			}
			resp.Outputs = append(resp.Outputs, buildOutputResult(out.Name, oc, result, nil, slot))
		}
		if err := fn(resp); err != nil {
			return err
		}
	}
	return nil
}

func (inst *Instance) close() {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.closed {
		return
	}
	inst.closed = true
	// Drain and close all pooled slots
	close(inst.slots)
	for slot := range inst.slots {
		slot.run.Close()
	}
	if inst.Session != nil {
		inst.Session.Close()
	}
	if inst.sam3 != nil {
		inst.sam3mu.Lock()
		inst.sam3.Close()
		inst.sam3mu.Unlock()
	}
}

func buildOutputResult(name string, oc *OutputConfig, result *onnx.Output, imgTensor *goimage.Tensor, slot *runSlot) *gomaxv1.OutputResult {
	or := &gomaxv1.OutputResult{
		Name:  name,
		Shape: result.Shape,
	}

	if oc == nil || result.Dtype != onnx.ElemTypeFloat32 || result.Len == 0 {
		or.Result = &gomaxv1.OutputResult_Raw{Raw: RawOutput(
			unsafe.Slice((*float32)(result.Ptr), result.Len), result.Shape,
		)}
		return or
	}

	data := unsafe.Slice((*float32)(result.Ptr), result.Len)

	switch oc.Type {
	case "detection":
		switch oc.Decoder {
		case "detr", "ov":
			// Both decoders read a separate normalized-box tensor named by oc.Boxes.
			boxes, err := slot.run.GetOutputByName(oc.Boxes)
			if err != nil || boxes.Dtype != onnx.ElemTypeFloat32 || boxes.Len == 0 {
				or.Result = &gomaxv1.OutputResult_Raw{Raw: RawOutput(data, result.Shape)}
				break
			}
			bdata := unsafe.Slice((*float32)(boxes.Ptr), boxes.Len)
			var dets *gomaxv1.Detections
			if oc.Decoder == "ov" {
				dets = OvDetrDetectOutput(data, result.Shape, bdata, oc.ResolvedLabels, imgTensor)
			} else {
				dets = DetrDetectOutput(data, result.Shape, bdata, oc.ResolvedLabels, imgTensor)
			}
			or.Result = &gomaxv1.OutputResult_Detections{Detections: dets}
		default: // "yolo" — boxes are packed inline in the single output tensor.
			or.Result = &gomaxv1.OutputResult_Detections{
				Detections: DetectOutput(data, result.Shape, oc.ResolvedLabels, imgTensor),
			}
		}
	case "segmentation":
		protos, err := slot.run.GetOutputByName(oc.Masks)
		if err != nil || imgTensor == nil || protos.Dtype != onnx.ElemTypeFloat32 || len(protos.Shape) != 4 || protos.Len == 0 {
			or.Result = &gomaxv1.OutputResult_Raw{Raw: RawOutput(data, result.Shape)}
			break
		}
		or.Result = &gomaxv1.OutputResult_Detections{Detections: SegmentOutput(
			data, result.Shape, unsafe.Slice((*float32)(protos.Ptr), protos.Len), protos.Shape,
			oc.ResolvedLabels, imgTensor)}
	case "classification":
		or.Result = &gomaxv1.OutputResult_Classifications{
			Classifications: ClassifyOutput(data, oc.ResolvedLabels, 10, oc.Sigmoid),
		}
	case "embedding":
		or.Result = &gomaxv1.OutputResult_Embedding{
			Embedding: EmbeddingOutput(data, result.Shape),
		}
	default:
		or.Result = &gomaxv1.OutputResult_Raw{Raw: RawOutput(data, result.Shape)}
	}

	return or
}

// withoutPrompt drops a "prompt" input the model does not have, so clients
// can send prompts without knowing whether the model uses them.
func withoutPrompt(cfg *ModelConfig, inputs []*gomaxv1.Input) []*gomaxv1.Input {
	if _, ok := cfg.Inputs[sam3PromptInput]; ok {
		return inputs
	}
	out := inputs[:0:0]
	for _, in := range inputs {
		if in.Name != sam3PromptInput {
			out = append(out, in)
		}
	}
	return out
}

func findInput(inputs []*gomaxv1.Input, name string) *gomaxv1.Input {
	for _, in := range inputs {
		if in.Name == name {
			return in
		}
	}
	if len(inputs) == 1 {
		return inputs[0]
	}
	return nil
}

func rawTensorToFloat32(t *gomaxv1.RawTensor) []float32 {
	if t == nil || len(t.Data) == 0 {
		return nil
	}
	n := len(t.Data) / 4
	out := make([]float32, n)
	for i := range n {
		bits := uint32(t.Data[i*4]) | uint32(t.Data[i*4+1])<<8 |
			uint32(t.Data[i*4+2])<<16 | uint32(t.Data[i*4+3])<<24
		out[i] = *(*float32)(unsafe.Pointer(&bits))
	}
	return out
}
