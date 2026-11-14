// Package sam3 runs the Segment Anything 3 (SAM 3) open-vocabulary pipeline as
// three chained ONNX models: an image encoder, a CLIP-style language encoder,
// and a mask/box decoder. Given an image and one or more text prompts it
// returns boxes, scores, and per-detection masks.
//
// Dataflow (per the upstream/AnyLabeling reference wiring):
//
//	image (uint8 [3,S,S]) ── image_encoder ──▶ vision_pos_enc_2, backbone_fpn_0/1/2
//	prompt text ── CLIP tokenize ──▶ tokens[1,32] ── language_encoder ──▶ mask, memory
//	         decoder(features, language, text-only box sentinels) ──▶ boxes, scores, masks
package sam3

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/aep/moxgo/pkg/clip"
	goimage "github.com/aep/moxgo/pkg/image"
	"github.com/aep/moxgo/pkg/onnx"
	"gopkg.in/yaml.v3"
)

// Detection is one decoded result for a single text prompt.
type Detection struct {
	Label string     // the prompt that produced it
	Score float32    // confidence
	Box   [4]float32 // raw decoder box (4 floats, as emitted)
	Mask  *Mask      // segmentation mask at original image resolution (nil if unavailable)
}

// Mask is a binary mask at original image resolution.
type Mask struct {
	W, H int
	Bits []bool // row-major, len == W*H
}

// At reports whether the mask is set at (x, y).
func (m *Mask) At(x, y int) bool {
	if x < 0 || y < 0 || x >= m.W || y >= m.H {
		return false
	}
	return m.Bits[y*m.W+x]
}

// Model holds the three loaded SAM 3 sessions and the tokenizer.
type Model struct {
	rt        *onnx.Runtime
	enc       *onnx.Session
	lang      *onnx.Session
	dec       *onnx.Session
	tok       *clip.Tokenizer
	inputSize int
	ActiveEP  string
}

// Options configures session loading.
type Options struct {
	// EP overrides the execution provider (e.g. "CUDA", "CPU"). Empty auto-selects.
	EP string
	// Threads sets intra-op threads (default 1).
	Threads int
}

// Open loads the SAM 3 models named in dir/config.yaml (or the conventional
// sam3_image_encoder/language_encoder/decoder.onnx filenames).
func Open(rt *onnx.Runtime, dir string, opts Options) (*Model, error) {
	encPath := filepath.Join(dir, "sam3_image_encoder.onnx")
	langPath := filepath.Join(dir, "sam3_language_encoder.onnx")
	decPath := filepath.Join(dir, "sam3_decoder.onnx")
	inputSize := 1008

	// Honor config.yaml if present.
	if cfg, err := loadConfig(filepath.Join(dir, "config.yaml")); err == nil {
		if cfg.EncoderModelPath != "" {
			encPath = resolve(dir, cfg.EncoderModelPath)
		}
		if cfg.LanguageEncoderPath != "" {
			langPath = resolve(dir, cfg.LanguageEncoderPath)
		}
		if cfg.DecoderModelPath != "" {
			decPath = resolve(dir, cfg.DecoderModelPath)
		}
		if cfg.InputSize > 0 {
			inputSize = cfg.InputSize
		}
	}

	threads := opts.Threads
	if threads <= 0 {
		threads = 1
	}
	eps := selectEPs(rt, opts.EP)
	sopts := onnx.SessionOptions{ExecutionProviders: eps, IntraOpThreads: threads, LogLevel: onnx.OrtLoggingLevelError}

	m := &Model{rt: rt, tok: clip.Default(), inputSize: inputSize}

	var err error
	if m.enc, err = rt.OpenSessionWith(encPath, sopts); err != nil {
		return nil, fmt.Errorf("sam3: open image encoder: %w (%s)", err, rt.LastError())
	}
	if m.lang, err = rt.OpenSessionWith(langPath, sopts); err != nil {
		m.Close()
		return nil, fmt.Errorf("sam3: open language encoder: %w (%s)", err, rt.LastError())
	}
	if m.dec, err = rt.OpenSessionWith(decPath, sopts); err != nil {
		m.Close()
		return nil, fmt.Errorf("sam3: open decoder: %w (%s)", err, rt.LastError())
	}
	m.ActiveEP = m.enc.ActiveEP
	return m, nil
}

// Close releases all sessions.
func (m *Model) Close() {
	if m.enc != nil {
		m.enc.Close()
	}
	if m.lang != nil {
		m.lang.Close()
	}
	if m.dec != nil {
		m.dec.Close()
	}
}

// Detect runs the full pipeline: encodes the image once, then runs the language
// encoder + decoder for each prompt. Detections with score <= confThreshold are
// dropped.
func (m *Model) Detect(img image.Image, prompts []string, confThreshold float32) ([]Detection, error) {
	imgData, srcW, srcH := goimage.FromImageUint8CHW(img, m.inputSize, m.inputSize)

	// --- Stage 1: image encoder (once) ---
	encRun := m.enc.NewRun()
	defer encRun.Close()
	if err := onnx.BindSlice(encRun, m.enc.Inputs[0].Name, imgData,
		[]int64{3, int64(m.inputSize), int64(m.inputSize)}); err != nil {
		return nil, fmt.Errorf("sam3: bind image: %w", err)
	}
	if err := encRun.Exec(); err != nil {
		return nil, fmt.Errorf("sam3: image encoder: %w (%s)", err, m.rt.LastError())
	}
	feats, err := collectByName(encRun, "vision_pos_enc_2", "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2")
	if err != nil {
		return nil, fmt.Errorf("sam3: image features: %w", err)
	}

	// Stable scalar/prompt buffers shared across prompts.
	origH := []int64{int64(srcH)}
	origW := []int64{int64(srcW)}
	boxCoords := []float32{0, 0, 0, 0}
	boxLabels := []int64{1}
	boxMasks := []bool{true} // true = text-only (no real geometric box)

	var out []Detection
	for _, prompt := range prompts {
		dets, err := m.detectOne(encRun, feats, prompt, origH, origW, boxCoords, boxLabels, boxMasks, srcW, srcH, confThreshold)
		if err != nil {
			return nil, err
		}
		out = append(out, dets...)
	}
	return out, nil
}

func (m *Model) detectOne(encRun *onnx.Run, feats map[string]*onnx.Output, prompt string,
	origH, origW []int64, boxCoords []float32, boxLabels []int64, boxMasks []bool,
	srcW, srcH int, confThreshold float32) ([]Detection, error) {

	// --- Stage 2: language encoder ---
	tokens := m.tok.Tokenize(prompt, clip.ContextLength)
	langRun := m.lang.NewRun()
	defer langRun.Close()
	if err := onnx.BindSlice(langRun, "tokens", tokens, []int64{1, clip.ContextLength}); err != nil {
		return nil, fmt.Errorf("sam3: bind tokens: %w", err)
	}
	if err := langRun.Exec(); err != nil {
		return nil, fmt.Errorf("sam3: language encoder: %w (%s)", err, m.rt.LastError())
	}
	langMask, err := langRun.GetOutputByName("text_attention_mask")
	if err != nil {
		return nil, fmt.Errorf("sam3: text_attention_mask: %w", err)
	}
	langFeat, err := langRun.GetOutputByName("text_memory")
	if err != nil {
		return nil, fmt.Errorf("sam3: text_memory: %w", err)
	}

	// --- Stage 3: decoder ---
	decRun := m.dec.NewRun()
	defer decRun.Close()

	bind := func(name string, o *onnx.Output) error {
		return decRun.Bind(name, o.Ptr, o.Len*o.Dtype.Size(), o.Dtype, o.Shape)
	}
	if err := onnx.BindSlice(decRun, "original_height", origH, []int64{}); err != nil {
		return nil, err
	}
	if err := onnx.BindSlice(decRun, "original_width", origW, []int64{}); err != nil {
		return nil, err
	}
	for name, o := range feats {
		if err := bind(name, o); err != nil {
			return nil, fmt.Errorf("sam3: bind %s: %w", name, err)
		}
	}
	if err := bind("language_mask", langMask); err != nil {
		return nil, err
	}
	if err := bind("language_features", langFeat); err != nil {
		return nil, err
	}
	if err := onnx.BindSlice(decRun, "box_coords", boxCoords, []int64{1, 1, 4}); err != nil {
		return nil, err
	}
	if err := onnx.BindSlice(decRun, "box_labels", boxLabels, []int64{1, 1}); err != nil {
		return nil, err
	}
	if err := onnx.BindBool(decRun, "box_masks", boxMasks, []int64{1, 1}); err != nil {
		return nil, err
	}

	if err := decRun.Exec(); err != nil {
		return nil, fmt.Errorf("sam3: decoder: %w (%s)", err, m.rt.LastError())
	}

	return decodeOutputs(decRun, prompt, confThreshold)
}

// decodeOutputs reads boxes/scores/masks and filters by confidence.
func decodeOutputs(r *onnx.Run, label string, confThreshold float32) ([]Detection, error) {
	scoresOut, err := r.GetOutputByName("scores")
	if err != nil {
		return nil, fmt.Errorf("sam3: scores: %w", err)
	}
	scores := unsafeFloat(scoresOut)
	boxesOut, err := r.GetOutputByName("boxes")
	if err != nil {
		return nil, fmt.Errorf("sam3: boxes: %w", err)
	}
	boxes := unsafeFloat(boxesOut)

	masksOut, masksErr := r.GetOutputByName("masks")
	var maskBits []bool
	var mh, mw int
	if masksErr == nil && masksOut.Dtype == onnx.ElemTypeBool && len(masksOut.Shape) == 4 {
		maskBits = unsafeBool(masksOut)
		mh = int(masksOut.Shape[2])
		mw = int(masksOut.Shape[3])
	}

	n := len(scores)
	out := make([]Detection, 0, n)
	for i := 0; i < n; i++ {
		if scores[i] <= confThreshold {
			continue
		}
		d := Detection{Label: label, Score: scores[i]}
		if i*4+4 <= len(boxes) {
			copy(d.Box[:], boxes[i*4:i*4+4])
		}
		if maskBits != nil {
			per := mh * mw
			if (i+1)*per <= len(maskBits) {
				bits := make([]bool, per) // copy: source freed on next Exec/Close
				copy(bits, maskBits[i*per:(i+1)*per])
				d.Mask = &Mask{W: mw, H: mh, Bits: bits}
			}
		}
		out = append(out, d)
	}
	return out, nil
}

// --- helpers ---

func collectByName(r *onnx.Run, names ...string) (map[string]*onnx.Output, error) {
	m := make(map[string]*onnx.Output, len(names))
	for _, n := range names {
		o, err := r.GetOutputByName(n)
		if err != nil {
			return nil, err
		}
		// GetOutput reuses an internal buffer per index, so capture a stable copy
		// of the struct (the underlying ORT data stays valid until Exec/Close).
		cp := *o
		m[n] = &cp
	}
	return m, nil
}

func unsafeFloat(o *onnx.Output) []float32 {
	if o == nil || o.Dtype != onnx.ElemTypeFloat32 || o.Len == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(o.Ptr), o.Len)
}

func unsafeBool(o *onnx.Output) []bool {
	if o == nil || o.Len == 0 {
		return nil
	}
	return unsafe.Slice((*bool)(o.Ptr), o.Len)
}

func selectEPs(rt *onnx.Runtime, ep string) []onnx.ExecutionProvider {
	if ep != "" {
		if strings.EqualFold(ep, "cpu") {
			return nil
		}
		return []onnx.ExecutionProvider{{Name: ep}}
	}
	var eps []onnx.ExecutionProvider
	if providers, err := rt.GetAvailableProviders(); err == nil {
		for _, p := range providers {
			if p != "CPUExecutionProvider" {
				eps = append(eps, onnx.ExecutionProvider{Name: strings.TrimSuffix(p, "ExecutionProvider")})
			}
		}
	}
	return eps
}

func resolve(dir, p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(dir, p)
}

type config struct {
	EncoderModelPath    string `yaml:"encoder_model_path"`
	DecoderModelPath    string `yaml:"decoder_model_path"`
	LanguageEncoderPath string `yaml:"language_encoder_path"`
	InputSize           int    `yaml:"input_size"`
}

func loadConfig(path string) (*config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c config
	if err := yaml.Unmarshal(data, &c); err != nil {
		return nil, err
	}
	return &c, nil
}
