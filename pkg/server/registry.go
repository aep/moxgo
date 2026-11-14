package server

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/aep/moxgo/pkg/labels"
	"gopkg.in/yaml.v3"
)

// InputConfig describes preprocessing for a model input.
type InputConfig struct {
	Name       string    `yaml:"-" json:"-"`
	Type       string    `yaml:"type" json:"type"`
	Width      int       `yaml:"width" json:"width,omitempty"`
	Height     int       `yaml:"height" json:"height,omitempty"`
	Mean       []float32 `yaml:"mean" json:"mean,omitempty"`
	Std        []float32 `yaml:"std" json:"std,omitempty"`
	SampleRate int       `yaml:"sample_rate" json:"sample_rate,omitempty"`
	Chunk      float64   `yaml:"chunk" json:"chunk,omitempty"`
	Overlap    float64   `yaml:"overlap" json:"overlap,omitempty"`
}

func (ic *InputConfig) InputSize() int {
	switch ic.Type {
	case "image":
		return 3 * ic.Width * ic.Height
	case "audio":
		return int(ic.Chunk * float64(ic.SampleRate))
	default:
		return 0
	}
}

func (ic *InputConfig) InputShape() []int64 {
	switch ic.Type {
	case "image":
		return []int64{1, 3, int64(ic.Height), int64(ic.Width)}
	case "audio":
		return []int64{1, int64(ic.Chunk * float64(ic.SampleRate))}
	default:
		return nil
	}
}

func (ic *InputConfig) ParamsMap() map[string]string {
	m := make(map[string]string)
	switch ic.Type {
	case "image":
		m["width"] = strconv.Itoa(ic.Width)
		m["height"] = strconv.Itoa(ic.Height)
	case "audio":
		m["sample_rate"] = strconv.Itoa(ic.SampleRate)
		m["chunk"] = strconv.FormatFloat(ic.Chunk, 'f', -1, 64)
		if ic.Overlap > 0 {
			m["overlap"] = strconv.FormatFloat(ic.Overlap, 'f', -1, 64)
		}
	}
	return m
}

// OutputConfig describes a model output.
type OutputConfig struct {
	Name    string  `yaml:"-" json:"-"`
	Type    string  `yaml:"type" json:"type"`
	Decoder string  `yaml:"decoder" json:"decoder,omitempty"`
	Labels  string  `yaml:"labels" json:"labels,omitempty"`
	Sigmoid float64 `yaml:"sigmoid" json:"sigmoid,omitempty"`
	Boxes   string  `yaml:"boxes" json:"boxes,omitempty"`
	// Masks names the mask prototype tensor of a segmentation output, e.g.
	// a YOLO -seg model's [1, 32, H/4, W/4] "output1".
	Masks          string        `yaml:"masks" json:"masks,omitempty"`
	ResolvedLabels labels.Labels `yaml:"-" json:"-"`
}

// ModelConfig defines a model in the registry.
type ModelConfig struct {
	Name string `yaml:"-"`
	Path string `yaml:"path"`
	EP   string `yaml:"ep"`
	// Needs are capabilities the model needs of its runtime; see
	// SetEPResolver.
	Needs   []string                `yaml:"needs"`
	Threads int                     `yaml:"threads"`
	Inputs  map[string]*InputConfig `yaml:"inputs"`

	// Pipeline selects a built-in multi-model pipeline instead of a single
	// ONNX graph. "sam3": path is a directory with the SAM 3 image encoder,
	// language encoder and decoder (see pkg/sam3); inputs/outputs are fixed.
	Pipeline string `yaml:"pipeline"`
	// Prompts are the default text prompts of a sam3 pipeline, used when a
	// request has no "prompt" input.
	Prompts []string `yaml:"prompts"`
	// Threshold is the minimum detection score of a sam3 pipeline (default 0.5).
	Threshold float32                  `yaml:"threshold"`
	Outputs   map[string]*OutputConfig `yaml:"outputs"`

	// Ordered slices built after parsing for deterministic iteration.
	InputList  []*InputConfig  `yaml:"-"`
	OutputList []*OutputConfig `yaml:"-"`
}

type registryFile struct {
	Models map[string]*ModelConfig `yaml:"models"`
}

// Registry holds validated model configurations. Safe for concurrent use:
// models can be added while the server runs.
type Registry struct {
	mu     sync.RWMutex
	models map[string]*ModelConfig
	order  []string
}

func NewRegistry() *Registry {
	return &Registry{models: make(map[string]*ModelConfig)}
}

// Add adds a model unless one of that name is already known.
func (r *Registry) Add(cfg *ModelConfig) error {
	r.mu.RLock()
	_, exists := r.models[cfg.Name]
	r.mu.RUnlock()
	if exists {
		return nil
	}
	return r.put(cfg, false)
}

// Replace adds a model, replacing one of the same name.
func (r *Registry) Replace(cfg *ModelConfig) error {
	return r.put(cfg, true)
}

func (r *Registry) put(cfg *ModelConfig, replace bool) error {
	cfg.InputList, cfg.OutputList = nil, nil
	for k, v := range cfg.Inputs {
		v.Name = k
		cfg.InputList = append(cfg.InputList, v)
	}
	for k, v := range cfg.Outputs {
		v.Name = k
		cfg.OutputList = append(cfg.OutputList, v)
	}

	if err := validateModel(cfg); err != nil {
		return fmt.Errorf("model %q: %w", cfg.Name, err)
	}

	for oname, oc := range cfg.Outputs {
		if oc.Labels != "" {
			var err error
			oc.ResolvedLabels, err = resolveLabels(oc.Labels)
			if err != nil {
				return fmt.Errorf("model %q: output %q: labels: %w", cfg.Name, oname, err)
			}
		}
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.models[cfg.Name]; exists {
		if !replace {
			return nil
		}
	} else {
		r.order = append(r.order, cfg.Name)
	}
	r.models[cfg.Name] = cfg
	return nil
}

// LoadRegistry reads and validates a YAML config file.
func LoadRegistry(path string) (*Registry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("registry: %w", err)
	}

	var rf registryFile
	if err := yaml.Unmarshal(data, &rf); err != nil {
		return nil, fmt.Errorf("registry: parse %s: %w", path, err)
	}

	// Relative model paths are resolved against the config file's directory,
	// not the current working directory.
	baseDir := filepath.Dir(path)

	reg := &Registry{
		models: make(map[string]*ModelConfig, len(rf.Models)),
	}

	for name, cfg := range rf.Models {
		cfg.Name = name
		if cfg.Path != "" && !filepath.IsAbs(cfg.Path) {
			cfg.Path = filepath.Join(baseDir, cfg.Path)
		}

		// Set names from map keys and build ordered lists
		for k, v := range cfg.Inputs {
			v.Name = k
			cfg.InputList = append(cfg.InputList, v)
		}
		for k, v := range cfg.Outputs {
			v.Name = k
			cfg.OutputList = append(cfg.OutputList, v)
		}

		if err := validateModel(cfg); err != nil {
			return nil, fmt.Errorf("registry: model %q: %w", name, err)
		}

		// Resolve labels per output
		for oname, oc := range cfg.Outputs {
			if oc.Labels != "" {
				oc.ResolvedLabels, err = resolveLabels(oc.Labels)
				if err != nil {
					return nil, fmt.Errorf("registry: model %q: output %q: labels: %w", name, oname, err)
				}
			}
		}

		reg.models[name] = cfg
		reg.order = append(reg.order, name)
	}

	return reg, nil
}

// ModelName is name without a default ":latest" variant tag: "yolo11n"
// and "yolo11n:latest" are the same model.
func ModelName(name string) string {
	return strings.TrimSuffix(name, ":latest")
}

func (r *Registry) Get(name string) (*ModelConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.models[ModelName(name)]
	return m, ok
}

func (r *Registry) List() []*ModelConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*ModelConfig, len(r.order))
	for i, name := range r.order {
		out[i] = r.models[name]
	}
	return out
}

func validateModel(cfg *ModelConfig) error {
	if cfg.Path == "" {
		return fmt.Errorf("path is required")
	}
	if _, err := os.Stat(cfg.Path); err != nil {
		return fmt.Errorf("path %q: %w", cfg.Path, err)
	}
	switch cfg.Pipeline {
	case "":
	case "sam3":
		return validateSAM3(cfg)
	default:
		return fmt.Errorf("unknown pipeline %q (want sam3)", cfg.Pipeline)
	}
	if len(cfg.Inputs) == 0 {
		return fmt.Errorf("at least one input is required")
	}
	for name, inp := range cfg.Inputs {
		switch inp.Type {
		case "image":
			if inp.Width <= 0 || inp.Height <= 0 {
				return fmt.Errorf("input %q: width and height are required for image", name)
			}
		case "audio":
			if inp.SampleRate <= 0 {
				return fmt.Errorf("input %q: sample_rate is required for audio", name)
			}
			if inp.Chunk <= 0 {
				return fmt.Errorf("input %q: chunk is required for audio", name)
			}
		case "raw":
		default:
			return fmt.Errorf("input %q: unknown type %q (want image, audio, or raw)", name, inp.Type)
		}
	}
	for name, out := range cfg.Outputs {
		switch out.Type {
		case "classification", "detection", "embedding", "raw":
		case "segmentation":
			// YOLO -seg: detections, each with mask coefficients after its
			// class scores, and a separate tensor of mask prototypes.
			if out.Decoder == "" {
				out.Decoder = "yolo"
			}
			if out.Decoder != "yolo" {
				return fmt.Errorf("output %q: unknown segmentation decoder %q (want yolo)", name, out.Decoder)
			}
			if out.Masks == "" {
				return fmt.Errorf("output %q: segmentation requires masks: <prototype tensor name>", name)
			}
		default:
			return fmt.Errorf("output %q: unknown type %q (want classification, detection, segmentation, embedding, or raw)", name, out.Type)
		}
		if out.Type == "detection" {
			if out.Decoder == "" {
				// Back-compat: infer the decoder from whether a separate box
				// tensor was named (DETR), otherwise assume packed YOLO output.
				if out.Boxes != "" {
					out.Decoder = "detr"
				} else {
					out.Decoder = "yolo"
				}
			}
			switch out.Decoder {
			case "yolo":
			case "detr", "ov":
				if out.Boxes == "" {
					return fmt.Errorf("output %q: decoder %q requires boxes: <tensor name>", name, out.Decoder)
				}
			default:
				return fmt.Errorf("output %q: unknown decoder %q (want yolo, detr, or ov)", name, out.Decoder)
			}
		}
	}
	return nil
}

func resolveLabels(name string) (labels.Labels, error) {
	switch name {
	case "coco80":
		return labels.COCO80, nil
	case "coco91":
		return labels.COCO91, nil
	default:
		return labels.Load(name)
	}
}
