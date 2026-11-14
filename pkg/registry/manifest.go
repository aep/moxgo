package registry

import "github.com/aep/moxgo/pkg/server"

type Manifest struct {
	Description string                          `json:"description,omitempty"`
	License     string                          `json:"license,omitempty"`
	Files       []File                          `json:"files"`
	Inputs      map[string]*server.InputConfig  `json:"inputs"`
	Outputs     map[string]*server.OutputConfig `json:"outputs,omitempty"`
	EP          string                          `json:"ep,omitempty"`
	// Needs are capabilities the model needs of its runtime, which runtime
	// rules can match (see Runtimes).
	Needs   []string `json:"needs,omitempty"`
	Threads int      `json:"threads,omitempty"`
}

type File struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

type IndexEntry struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	License     string `json:"license"`
	Size        int64  `json:"size"`
	// Variants are the model's other variants, pulled as "name:tag".
	Variants []Variant `json:"variants,omitempty"`
}

// Variant is a variant of a model other than its default, e.g. a
// quantized one.
type Variant struct {
	Tag         string `json:"tag"`
	Description string `json:"description"`
	Size        int64  `json:"size"`
}

type Index struct {
	Models []IndexEntry `json:"models"`
}
