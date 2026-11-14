// Package local builds an in-process moxgo: the ONNX runtime, the model
// registry (an optional config file plus the models pulled to
// ~/.moxgo/models), and pulling from the public registry. It is what
// `moxgo serve` and `moxgo run` use, and what a program embedding moxgo
// uses instead of talking to a server.
package local

import (
	"fmt"
	"io"
	"os"

	"github.com/aep/moxgo/pkg/onnx"
	"github.com/aep/moxgo/pkg/registry"
	"github.com/aep/moxgo/pkg/server"
)

// Options for New. The zero value auto-detects the ONNX runtime and uses
// pulled models only.
type Options struct {
	// LibPath is the ONNX runtime shared library. Empty: platform defaults.
	LibPath string

	// ConfigPath is a model registry YAML file (moxgo.yaml). Empty or
	// missing: none.
	ConfigPath string

	// Warnings receives problems with single models that do not stop the
	// rest, e.g. a broken pulled model. nil: os.Stderr.
	Warnings io.Writer
}

// New loads the ONNX runtime and the models. The caller must call Close on
// the returned server's runtime via Close when done.
func New(opts Options) (*server.Server, *onnx.Runtime, error) {
	var rt *onnx.Runtime
	var err error
	if opts.LibPath != "" {
		rt, err = onnx.LoadFrom(opts.LibPath)
	} else {
		rt, err = onnx.Load()
	}
	if err != nil {
		return nil, nil, fmt.Errorf("load ONNX Runtime: %w", err)
	}
	warn := opts.Warnings
	if warn == nil {
		warn = os.Stderr
	}

	reg := server.NewRegistry()
	if opts.ConfigPath != "" {
		if _, statErr := os.Stat(opts.ConfigPath); statErr == nil {
			reg, err = server.LoadRegistry(opts.ConfigPath)
			if err != nil {
				rt.Close()
				return nil, nil, err
			}
		}
	}

	if localModels, err := registry.ListLocal(); err == nil {
		for _, m := range localModels {
			if err := reg.Add(m); err != nil {
				_, _ = fmt.Fprintf(warn, "  warning: %v\n", err)
			}
		}
	}

	return server.New(rt, reg, Pull), rt, nil
}

// Pull downloads a model from the public registry to ~/.moxgo/models.
func Pull(name string, progress func(string, int64, int64)) (*server.ModelConfig, error) {
	cb := func(p registry.Progress) {
		if progress != nil {
			progress(p.File, p.Total, p.Downloaded)
		}
	}
	if _, err := registry.Pull(registry.DefaultRegistry, name, cb); err != nil {
		return nil, err
	}
	dir, err := registry.ModelDir(name)
	if err != nil {
		return nil, err
	}
	return registry.LoadDir(name, dir)
}
