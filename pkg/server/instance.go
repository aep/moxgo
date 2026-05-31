package server

import (
	"bytes"
	"fmt"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/aep/moxgo/pkg/audio"
	goimage "github.com/aep/moxgo/pkg/image"
	"github.com/aep/moxgo/pkg/labels"
	"github.com/aep/moxgo/pkg/onnx"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
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
}

// InstanceManager manages running model instances.
type InstanceManager struct {
	rt       *onnx.Runtime
	registry *Registry

	mu        sync.RWMutex
	instances map[string]*Instance
}

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
		threads = 1
	}

	var eps []onnx.ExecutionProvider
	if ep != "" {
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

	m.mu.Lock()
	// Check again under write lock (race)
	if existing, ok := m.instances[modelName]; ok {
		m.mu.Unlock()
		sess.Close()
		return existing, nil
	}
	m.instances[modelName] = inst
	m.mu.Unlock()

	return inst, nil
}

// GetOrRun returns a running instance, auto-loading if needed.
func (m *InstanceManager) GetOrRun(modelName string) (*Instance, error) {
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
	inst.mu.Lock()
	if inst.closed {
		inst.mu.Unlock()
		return nil, fmt.Errorf("model %s is stopped", inst.Name)
	}
	inst.mu.Unlock()

	slot := inst.acquireSlot()
	defer inst.releaseSlot(slot)

	var imgTensors []*goimage.Tensor
	for i, inp := range inst.Config.InputList {
		reqInput := findInput(inputs, inp.Name)
		if reqInput == nil {
			return nil, fmt.Errorf("missing input %q", inp.Name)
		}

		switch inp.Type {
		case "image":
			tensor, err := goimage.Decode(bytes.NewReader(reqInput.GetFileBytes()), inp.Width, inp.Height)
			if err != nil {
				return nil, fmt.Errorf("decode image %q: %w", inp.Name, err)
			}
			copy(slot.bufs[i], tensor.Data)
			imgTensors = append(imgTensors, tensor)

		case "audio":
			at, err := audio.Decode(bytes.NewReader(reqInput.GetFileBytes()), audio.Options{
				SampleRate: inp.SampleRate,
				MaxSamples: inp.InputSize(),
			})
			if err != nil {
				return nil, fmt.Errorf("decode audio %q: %w", inp.Name, err)
			}
			n := copy(slot.bufs[i], at.Data)
			for j := n; j < len(slot.bufs[i]); j++ {
				slot.bufs[i][j] = 0
			}

		case "raw":
			rawData := rawTensorToFloat32(reqInput.GetTensor())
			if len(rawData) != len(slot.bufs[i]) {
				return nil, fmt.Errorf("input %q: size %d != expected %d", inp.Name, len(rawData), len(slot.bufs[i]))
			}
			copy(slot.bufs[i], rawData)
		}
	}

	if !slot.warmed {
		if err := inst.warmSlot(slot); err != nil {
			return nil, err
		}
	}

	start := time.Now()
	if err := slot.run.Exec(); err != nil {
		return nil, fmt.Errorf("exec: %w", err)
	}
	elapsed := time.Since(start)

	var firstImg *goimage.Tensor
	if len(imgTensors) > 0 {
		firstImg = imgTensors[0]
	}

	resp := &gomaxv1.PredictResponse{
		InferenceTimeMs: float64(elapsed.Microseconds()) / 1000.0,
	}
	for i, out := range inst.Session.Outputs {
		result, err := slot.run.GetOutput(i)
		if err != nil {
			continue
		}
		resp.Outputs = append(resp.Outputs, buildOutputResult(out.Name, inst.Config.Outputs[out.Name], result, firstImg))
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

	if len(inst.Config.InputList) != 1 || inst.Config.InputList[0].Type != "audio" {
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
			result, err := slot.run.GetOutput(i)
			if err != nil {
				continue
			}
			resp.Outputs = append(resp.Outputs, buildOutputResult(out.Name, inst.Config.Outputs[out.Name], result, nil))
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
	inst.Session.Close()
}

func buildOutputResult(name string, oc *OutputConfig, result *onnx.Output, imgTensor *goimage.Tensor) *gomaxv1.OutputResult {
	or := &gomaxv1.OutputResult{
		Name:  name,
		Shape: result.Shape,
	}

	if result.Dtype != onnx.ElemTypeFloat32 || result.Len == 0 {
		or.Result = &gomaxv1.OutputResult_Raw{Raw: RawOutput(
			unsafe.Slice((*float32)(result.Ptr), result.Len), result.Shape,
		)}
		return or
	}

	data := unsafe.Slice((*float32)(result.Ptr), result.Len)
	var lbls labels.Labels
	if oc != nil {
		lbls = oc.ResolvedLabels
	}

	switch {
	case IsYOLODetectionShape(result.Shape) && imgTensor != nil && lbls != nil:
		or.Result = &gomaxv1.OutputResult_Detections{
			Detections: DetectOutput(data, result.Shape, lbls, imgTensor),
		}
	case IsClassificationShape(result.Shape) && lbls != nil:
		var sigmoid float64
		if oc != nil {
			sigmoid = oc.Sigmoid
		}
		or.Result = &gomaxv1.OutputResult_Classifications{
			Classifications: ClassifyOutput(data, lbls, 10, sigmoid),
		}
	default:
		or.Result = &gomaxv1.OutputResult_Raw{Raw: RawOutput(data, result.Shape)}
	}

	return or
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
