package local

import (
	"fmt"
	"io"
	"os"
	"sort"
	"sync"

	"github.com/aep/moxgo/pkg/onnx"
	"github.com/aep/moxgo/pkg/registry"
	"github.com/aep/moxgo/pkg/server"
)

// RuntimeOptions selects the ONNX Runtime by moxgo's runtime rules (see
// registry.Runtimes) from the packages a build bundled, and those the
// rules let moxgo download.
type RuntimeOptions struct {
	// Bundled is where the build put its runtime packages (see
	// registry.BundleRuntime), e.g. an Android app's native library
	// directory, which is Flat.
	Bundled registry.Layout

	// Downloads is where downloaded packages go. Empty:
	// registry.RuntimesDir().
	Downloads string

	// NoDownload uses bundled packages only.
	NoDownload bool

	// Runtimes are the rules. nil: registry.DefaultRuntimes().
	Runtimes *registry.Runtimes

	// Facts are added to the detected ones (registry.DetectFacts),
	// replacing those of the same name.
	Facts registry.Facts

	// Progress receives download progress. nil: none.
	Progress func(file string, total, downloaded int64)
}

// runtimeSelector loads the ONNX Runtime the rules give this device, and
// gives each model the execution providers the rules give it, downloading,
// preloading and registering what those need. The ONNX Runtime library is
// loaded once, from the rules without any model's needs and from bundled
// packages only: a process can have one, and starting needs no network.
// What a model's providers need is downloaded when the first model that
// needs it runs; if that fails, models run with what is bundled.
type runtimeSelector struct {
	opts      RuntimeOptions
	all       *registry.Runtimes // rules usable with downloads
	bundled   *registry.Runtimes // rules usable with bundled packages only
	facts     registry.Facts
	downloads registry.Layout
	warn      io.Writer

	mu      sync.Mutex
	rt      *onnx.Runtime
	core    string
	failed  bool            // a download failed: use bundled packages only
	loaded  map[string]bool // preloaded files, by ref
	envSet  map[string]string
	plugins map[string]string // registered plugins, name -> ref
}

func newRuntimeSelector(opts RuntimeOptions, warn io.Writer) (*runtimeSelector, error) {
	rs := opts.Runtimes
	if rs == nil {
		rs = registry.DefaultRuntimes()
	}
	if opts.Downloads == "" {
		dir, err := registry.RuntimesDir()
		if err != nil {
			return nil, err
		}
		opts.Downloads = dir
	}
	facts := registry.DetectFacts()
	for k, v := range opts.Facts {
		facts[k] = v
	}
	s := &runtimeSelector{
		opts: opts, facts: facts, warn: warn,
		downloads: registry.Layout{Root: opts.Downloads},
		loaded:    map[string]bool{}, envSet: map[string]string{}, plugins: map[string]string{},
	}
	s.bundled = rs.Available(func(pkg string) bool { return opts.Bundled.Has(rs, pkg) })
	s.all = rs.Available(func(pkg string) bool {
		return opts.Bundled.Has(rs, pkg) || (!opts.NoDownload && rs.Packages[pkg].Download)
	})
	return s, nil
}

// Dir places a package: bundled if the build has it, else downloaded.
func (s *runtimeSelector) Dir(pkg string) string {
	if s.opts.Bundled.Has(s.all, pkg) {
		return s.opts.Bundled.Dir(pkg)
	}
	return s.downloads.Dir(pkg)
}

// load loads the device's ONNX Runtime.
func (s *runtimeSelector) load() (*onnx.Runtime, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	res := s.bundled.Resolve(s.facts, nil)
	if res.Core == "" {
		return nil, fmt.Errorf("no runtime rule gives an ONNX Runtime bundled in %s for %s",
			s.opts.Bundled.Root, describeFacts(s.facts))
	}
	rt, err := onnx.LoadFrom(registry.PathIn(s, res.Core))
	if err != nil {
		return nil, err
	}
	s.rt, s.core = rt, res.Core
	return rt, nil
}

// eps resolves cfg's runtime and returns its execution providers. It is
// the server's EPResolver.
func (s *runtimeSelector) eps(cfg *server.ModelConfig) ([]onnx.ExecutionProvider, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rs := s.all
	if s.failed {
		rs = s.bundled
	}
	res := rs.Resolve(s.facts, cfg.Needs)
	if err := s.apply(res); err != nil {
		if rs == s.bundled {
			return nil, false, err
		}
		_, _ = fmt.Fprintf(s.warn, "  warning: %s: runtime: %v; using bundled runtimes only\n", cfg.Name, err)
		s.failed = true
		res = s.bundled.Resolve(s.facts, cfg.Needs)
		if err := s.apply(res); err != nil {
			return nil, false, err
		}
	}
	if res.Core != "" && res.Core != s.core {
		_, _ = fmt.Fprintf(s.warn, "  warning: %s: the rules give it ONNX Runtime %s, but %s is loaded\n",
			cfg.Name, res.Core, s.core)
	}
	if !res.HasEPs {
		return nil, false, nil
	}
	eps := make([]onnx.ExecutionProvider, 0, len(res.EPs))
	for _, ep := range res.EPs {
		opts := make(map[string]string, len(ep.Options))
		for k, v := range ep.Options {
			opts[k] = registry.ExpandIn(s, v)
		}
		eps = append(eps, onnx.ExecutionProvider{Name: ep.Name, Options: opts})
	}
	return eps, true, nil
}

// apply downloads what res needs and the build did not bundle, sets its
// environment, preloads its libraries and registers its plugins. Callers
// hold mu.
func (s *runtimeSelector) apply(res *registry.ResolvedRuntime) error {
	var fetch []string
	for _, pkg := range res.Packages() {
		if !s.opts.Bundled.Has(s.all, pkg) {
			fetch = append(fetch, pkg)
		}
	}
	if len(fetch) > 0 {
		var progress func(registry.Progress)
		if s.opts.Progress != nil {
			progress = func(p registry.Progress) { s.opts.Progress(p.File, p.Total, p.Downloaded) }
		}
		if err := registry.InstallRuntime(s.all, "", s.downloads.Root, fetch, progress); err != nil {
			return err
		}
	}

	keys := make([]string, 0, len(res.Env))
	for k := range res.Env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := registry.ExpandIn(s, res.Env[k])
		if old, ok := s.envSet[k]; ok {
			if old != v {
				_, _ = fmt.Fprintf(s.warn, "  warning: runtime: %s is already %s, not %s\n", k, old, v)
			}
			continue
		}
		if err := os.Setenv(k, v); err != nil {
			return err
		}
		s.envSet[k] = v
	}

	for _, ref := range res.Preload {
		if s.loaded[ref] {
			continue
		}
		if err := onnx.Preload(registry.PathIn(s, ref)); err != nil {
			return err
		}
		s.loaded[ref] = true
	}

	names := make([]string, 0, len(res.Plugins))
	for name := range res.Plugins {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		ref := res.Plugins[name]
		if old, ok := s.plugins[name]; ok {
			if old != ref {
				_, _ = fmt.Fprintf(s.warn, "  warning: runtime: execution provider %s is %s already, not %s\n", name, old, ref)
			}
			continue
		}
		if err := s.rt.RegisterEPLibrary(name, registry.PathIn(s, ref)); err != nil {
			return err
		}
		s.plugins[name] = ref
	}
	return nil
}

func describeFacts(f registry.Facts) string {
	keys := make([]string, 0, len(f))
	for k := range f {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := ""
	for i, k := range keys {
		if i > 0 {
			out += " "
		}
		out += k + "=" + f[k]
	}
	return out
}
