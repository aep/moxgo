package registry

import "github.com/aep/moxgo/pkg/server"

type Manifest struct {
	Description string                          `json:"description,omitempty"`
	License     string                          `json:"license,omitempty"`
	Files       []File                          `json:"files"`
	Inputs      map[string]*server.InputConfig  `json:"inputs"`
	Outputs     map[string]*server.OutputConfig `json:"outputs,omitempty"`
	EP          string                          `json:"ep,omitempty"`
	Threads     int                             `json:"threads,omitempty"`
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
}

type Index struct {
	Models []IndexEntry `json:"models"`
}
