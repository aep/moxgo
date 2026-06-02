package server

import (
	"fmt"
	"os"
	"strconv"

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
	Chunk      float64 `yaml:"chunk" json:"chunk,omitempty"`
	Overlap    float64 `yaml:"overlap" json:"overlap,omitempty"`
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
	Name           string        `yaml:"-" json:"-"`
	Type           string        `yaml:"type" json:"type"`
	Labels         string        `yaml:"labels" json:"labels,omitempty"`
	Sigmoid        float64       `yaml:"sigmoid" json:"sigmoid,omitempty"`
	Boxes          string        `yaml:"boxes" json:"boxes,omitempty"`
	ResolvedLabels labels.Labels `yaml:"-" json:"-"`
}

// ModelConfig defines a model in the registry.
type ModelConfig struct {
	Name    string                   `yaml:"-"`
	Path    string                   `yaml:"path"`
	EP      string                   `yaml:"ep"`
	Threads int                      `yaml:"threads"`
	Inputs  map[string]*InputConfig  `yaml:"inputs"`
	Outputs map[string]*OutputConfig `yaml:"outputs"`

	// Ordered slices built after parsing for deterministic iteration.
	InputList  []*InputConfig  `yaml:"-"`
	OutputList []*OutputConfig `yaml:"-"`
}

type registryFile struct {
	Models map[string]*ModelConfig `yaml:"models"`
}

// Registry holds validated model configurations.
type Registry struct {
	models map[string]*ModelConfig
	order  []string
}

func NewRegistry() *Registry {
	return &Registry{models: make(map[string]*ModelConfig)}
}

func (r *Registry) Add(cfg *ModelConfig) error {
	if _, exists := r.models[cfg.Name]; exists {
		return nil
	}

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

	r.models[cfg.Name] = cfg
	r.order = append(r.order, cfg.Name)
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

	reg := &Registry{
		models: make(map[string]*ModelConfig, len(rf.Models)),
	}

	for name, cfg := range rf.Models {
		cfg.Name = name

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

func (r *Registry) Get(name string) (*ModelConfig, bool) {
	m, ok := r.models[name]
	return m, ok
}

func (r *Registry) List() []*ModelConfig {
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
		default:
			return fmt.Errorf("output %q: unknown type %q (want classification, detection, embedding, or raw)", name, out.Type)
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
