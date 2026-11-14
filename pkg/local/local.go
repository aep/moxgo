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
	// LibPath is the ONNX runtime shared library. Empty: Runtime, else
	// platform defaults.
	LibPath string

	// Runtime, when set and LibPath is empty, loads the ONNX Runtime that
	// moxgo's runtime rules give this device from the packages the build
	// bundled, and gives each model the execution providers the rules give
	// it, downloading those packages the rules allow when first needed.
	Runtime *RuntimeOptions

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
	warn := opts.Warnings
	if warn == nil {
		warn = os.Stderr
	}
	var rt *onnx.Runtime
	var sel *runtimeSelector
	var err error
	switch {
	case opts.LibPath != "":
		rt, err = onnx.LoadFrom(opts.LibPath)
	case opts.Runtime != nil:
		sel, err = newRuntimeSelector(*opts.Runtime, warn)
		if err == nil {
			rt, err = sel.load()
		}
	default:
		rt, err = onnx.Load()
	}
	if err != nil {
		return nil, nil, fmt.Errorf("load ONNX Runtime: %w", err)
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

	srv := server.New(rt, reg, Pull)
	if sel != nil {
		srv.SetEPResolver(sel.eps)
	}
	return srv, rt, nil
}

// Pull downloads a model from the public registry to ~/.moxgo/models.
func Pull(name string, progress func(string, int64, int64)) (*server.ModelConfig, error) {
	cb := func(p registry.Progress) {
		if progress != nil {
			progress(p.File, p.Total, p.Downloaded)
		}
	}
	name = registry.CanonicalName(name)
	if _, err := registry.Pull(registry.DefaultRegistry, name, cb); err != nil {
		return nil, err
	}
	dir, err := registry.ModelDir(name)
	if err != nil {
		return nil, err
	}
	return registry.LoadDir(name, dir)
}
