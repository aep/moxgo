package registry

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Runtimes say which ONNX Runtime, plugin execution providers and support
// libraries to use where; moxgo has them built in (runtimes.json, see
// DefaultRuntimes). A build bundles the packages it ships (BundleRuntime);
// moxgo uses the rules that bundled packages, and those marked Download,
// satisfy, fetching the latter when a device first needs them. Packages are versioned
// sets of files; rules match a device by its facts (see Facts) and a model
// by what it needs, and each sets part of a runtime. All rules that match
// apply, least specific first, so a rule for Android can be refined by one
// for Qualcomm phones, then one for a chip, a phone, and a phone running a
// model that needs some capability.

// Runtimes is a runtimes.json.
type Runtimes struct {
	Packages map[string]*RuntimePackage `json:"packages"`
	Rules    []RuntimeRule              `json:"rules"`
}

// RuntimePackage is a set of files installed together into one directory.
type RuntimePackage struct {
	Description string `json:"description,omitempty"`
	License     string `json:"license,omitempty"`

	// Download lets moxgo fetch the package when a device needs it and the
	// build did not bundle it, e.g. a large or proprietary library that
	// only some devices use. Other packages are used only when bundled.
	Download bool `json:"download,omitempty"`

	// Source is an archive the files are members of. Optional: files may
	// instead have their own URL, or come from the registry's blobs.
	Source *Archive      `json:"source,omitempty"`
	Files  []RuntimeFile `json:"files"`
}

// Archive is a downloadable zip (also .aar, .jar, .whl) or .tar.gz.
type Archive struct {
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size,omitempty"`
	// Format is "zip" or "tar.gz". Empty: from the URL's extension.
	Format string `json:"format,omitempty"`
}

// RuntimeFile is one file of a package. It is the member Path of the
// package's Source, else downloaded from URL, else the registry blob of
// its SHA256.
type RuntimeFile struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size,omitempty"`
	Path   string `json:"path,omitempty"`
	URL    string `json:"url,omitempty"`
}

// RuntimeRule sets part of a runtime where it matches.
type RuntimeRule struct {
	Comment string `json:"comment,omitempty"`

	// Match maps fact names to patterns; the rule matches a device whose
	// facts match all of them. A pattern is a string or a list of strings,
	// any of which may match; strings are path.Match globs.
	Match map[string]Patterns `json:"match,omitempty"`

	// Needs are model capabilities; the rule only applies to models that
	// need all of them.
	Needs []string `json:"needs,omitempty"`

	Runtime RuntimeSpec `json:"runtime"`
}

// Patterns is one or more glob patterns; JSON may give a single string.
type Patterns []string

func (p *Patterns) UnmarshalJSON(b []byte) error {
	var one string
	if err := json.Unmarshal(b, &one); err == nil {
		*p = Patterns{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(b, &many); err != nil {
		return fmt.Errorf("match pattern: want a string or a list of strings: %w", err)
	}
	*p = many
	return nil
}

// RuntimeSpec is what a rule sets. Absent fields leave what less specific
// rules set; maps merge key by key, a null value removing the key; the
// other fields replace.
//
// Files are referenced as "package/file". Strings in Env and EP options
// may contain ${pkg:NAME}, the directory package NAME is installed in.
type RuntimeSpec struct {
	// Core is the ONNX Runtime library.
	Core *string `json:"core,omitempty"`

	// Plugins are plugin execution provider libraries, by the name EPs
	// select them with.
	Plugins map[string]*string `json:"plugins,omitempty"`

	// Preload are libraries loaded, in order, before the core: libraries
	// that others open by bare name, which the dynamic linker would not
	// find in the package directories (it matches already loaded ones).
	Preload *[]string `json:"preload,omitempty"`

	// Env are environment variables set before anything is loaded.
	Env map[string]*string `json:"env,omitempty"`

	// EPs are the execution providers to try, in order; the CPU is always
	// the fallback. Absent everywhere: moxgo picks from what the runtime
	// has. Empty: the CPU only.
	EPs *[]EPSpec `json:"eps,omitempty"`
}

// EPSpec is an execution provider and its options.
type EPSpec struct {
	Name    string            `json:"name"`
	Options map[string]string `json:"options,omitempty"`
}

// ResolvedRuntime is the runtime that rules give a device and model.
type ResolvedRuntime struct {
	Core    string
	Plugins map[string]string
	Preload []string
	Env     map[string]string
	EPs     []EPSpec
	// HasEPs is whether some rule set EPs; if not, EPs is nil and the
	// choice is moxgo's.
	HasEPs bool
	// Rules are the indices of the rules applied, in the order applied.
	Rules []int
}

// ParseRuntimes parses and checks a runtimes.json.
func ParseRuntimes(data []byte) (*Runtimes, error) {
	var rs Runtimes
	if err := json.Unmarshal(data, &rs); err != nil {
		return nil, fmt.Errorf("parse runtimes: %w", err)
	}
	if err := rs.check(); err != nil {
		return nil, err
	}
	return &rs, nil
}

var pkgVarRe = regexp.MustCompile(`\$\{pkg:([^}]+)\}`)

func (rs *Runtimes) check() error {
	for name, p := range rs.Packages {
		if name == "" || strings.ContainsAny(name, `/\`) || name == "." || name == ".." {
			return fmt.Errorf("runtimes: bad package name %q", name)
		}
		for _, f := range p.Files {
			if f.Name == "" || strings.ContainsAny(f.Name, `/\`) || f.Name == "." || f.Name == ".." {
				return fmt.Errorf("runtimes: package %s: bad file name %q", name, f.Name)
			}
			if f.SHA256 == "" {
				return fmt.Errorf("runtimes: package %s: file %s has no sha256", name, f.Name)
			}
			if f.Path != "" && p.Source == nil {
				return fmt.Errorf("runtimes: package %s: file %s has a path but the package no source", name, f.Name)
			}
		}
	}
	for i, r := range rs.Rules {
		for k, ps := range r.Match {
			for _, p := range ps {
				if _, err := path.Match(p, ""); err != nil {
					return fmt.Errorf("runtimes: rule %d: match %s: bad pattern %q", i, k, p)
				}
			}
		}
		spec := r.Runtime
		var refs []string
		if spec.Core != nil {
			refs = append(refs, *spec.Core)
		}
		for _, v := range spec.Plugins {
			if v != nil {
				refs = append(refs, *v)
			}
		}
		if spec.Preload != nil {
			refs = append(refs, *spec.Preload...)
		}
		for _, ref := range refs {
			if _, _, err := rs.file(ref); err != nil {
				return fmt.Errorf("runtimes: rule %d: %w", i, err)
			}
		}
		for _, s := range spec.strings() {
			for _, m := range pkgVarRe.FindAllStringSubmatch(s, -1) {
				if rs.Packages[m[1]] == nil {
					return fmt.Errorf("runtimes: rule %d: unknown package %q in %q", i, m[1], s)
				}
			}
		}
	}
	return nil
}

// strings are the spec's strings that may hold ${pkg:NAME}.
func (s RuntimeSpec) strings() []string {
	var out []string
	for _, v := range s.Env {
		if v != nil {
			out = append(out, *v)
		}
	}
	if s.EPs != nil {
		for _, ep := range *s.EPs {
			for _, v := range ep.Options {
				out = append(out, v)
			}
		}
	}
	return out
}

// file resolves a "package/file" reference.
func (rs *Runtimes) file(ref string) (string, *RuntimeFile, error) {
	pkg, name, ok := strings.Cut(ref, "/")
	if !ok {
		return "", nil, fmt.Errorf("file reference %q is not package/file", ref)
	}
	p := rs.Packages[pkg]
	if p == nil {
		return "", nil, fmt.Errorf("file reference %q: unknown package %q", ref, pkg)
	}
	for i := range p.Files {
		if p.Files[i].Name == name {
			return pkg, &p.Files[i], nil
		}
	}
	return "", nil, fmt.Errorf("file reference %q: package %s has no file %s", ref, pkg, name)
}

// Matches reports whether rule i applies to a device with facts running
// a model that needs needs.
func (rs *Runtimes) Matches(i int, facts Facts, needs []string) bool {
	r := rs.Rules[i]
	for k, ps := range r.Match {
		v, ok := facts[k]
		if !ok {
			return false
		}
		hit := false
		for _, p := range ps {
			if ok, _ := path.Match(p, v); ok {
				hit = true
				break
			}
		}
		if !hit {
			return false
		}
	}
	for _, n := range r.Needs {
		found := false
		for _, have := range needs {
			if have == n {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// Resolve applies the rules that match facts and needs, least specific
// first: a rule is more specific than another if it needs more model
// capabilities or, needing as many, matches more facts; of equally
// specific rules the later one in the file applies last.
func (rs *Runtimes) Resolve(facts Facts, needs []string) *ResolvedRuntime {
	var hits []int
	for i := range rs.Rules {
		if rs.Matches(i, facts, needs) {
			hits = append(hits, i)
		}
	}
	sort.SliceStable(hits, func(a, b int) bool {
		ra, rb := rs.Rules[hits[a]], rs.Rules[hits[b]]
		if len(ra.Needs) != len(rb.Needs) {
			return len(ra.Needs) < len(rb.Needs)
		}
		return len(ra.Match) < len(rb.Match)
	})

	res := &ResolvedRuntime{Plugins: map[string]string{}, Env: map[string]string{}, Rules: hits}
	for _, i := range hits {
		s := rs.Rules[i].Runtime
		if s.Core != nil {
			res.Core = *s.Core
		}
		for k, v := range s.Plugins {
			if v == nil {
				delete(res.Plugins, k)
			} else {
				res.Plugins[k] = *v
			}
		}
		if s.Preload != nil {
			res.Preload = append([]string(nil), *s.Preload...)
		}
		for k, v := range s.Env {
			if v == nil {
				delete(res.Env, k)
			} else {
				res.Env[k] = *v
			}
		}
		if s.EPs != nil {
			res.EPs = append([]EPSpec{}, *s.EPs...)
			res.HasEPs = true
		}
	}
	return res
}

// Packages are the packages the runtime uses, sorted.
func (r *ResolvedRuntime) Packages() []string {
	set := map[string]bool{}
	add := func(ref string) {
		if pkg, _, ok := strings.Cut(ref, "/"); ok {
			set[pkg] = true
		}
	}
	if r.Core != "" {
		add(r.Core)
	}
	for _, v := range r.Plugins {
		add(v)
	}
	for _, v := range r.Preload {
		add(v)
	}
	var strs []string
	for _, v := range r.Env {
		strs = append(strs, v)
	}
	for _, ep := range r.EPs {
		for _, v := range ep.Options {
			strs = append(strs, v)
		}
	}
	for _, s := range strs {
		for _, m := range pkgVarRe.FindAllStringSubmatch(s, -1) {
			set[m[1]] = true
		}
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Layout is where runtime packages are: each in Root/<package>, or with
// Flat all files directly in Root, as an Android app's native library
// directory has them.
type Layout struct {
	Root string
	Flat bool
}

// Locator says where packages are.
type Locator interface {
	Dir(pkg string) string
}

// PathIn is where the "package/file" ref is.
func PathIn(l Locator, ref string) string {
	pkg, name, _ := strings.Cut(ref, "/")
	return filepath.Join(l.Dir(pkg), name)
}

// ExpandIn replaces ${pkg:NAME} in s with NAME's directory.
func ExpandIn(l Locator, s string) string {
	return pkgVarRe.ReplaceAllStringFunc(s, func(m string) string {
		return l.Dir(pkgVarRe.FindStringSubmatch(m)[1])
	})
}

// Dir is package pkg's directory.
func (l Layout) Dir(pkg string) string {
	if l.Flat {
		return l.Root
	}
	return filepath.Join(l.Root, pkg)
}

// Has reports whether all of package pkg's files are there. It does not
// check their contents, which bundling did.
func (l Layout) Has(rs *Runtimes, pkg string) bool {
	p := rs.Packages[pkg]
	if p == nil {
		return false
	}
	for _, f := range p.Files {
		if _, err := os.Stat(filepath.Join(l.Dir(pkg), f.Name)); err != nil {
			return false
		}
	}
	return true
}

//go:embed runtimes.json
var defaultRuntimes []byte

// DefaultRuntimes are the runtimes built into moxgo.
func DefaultRuntimes() *Runtimes {
	rs, err := ParseRuntimes(defaultRuntimes)
	if err != nil {
		panic("moxgo: built-in runtimes.json: " + err.Error())
	}
	return rs
}

// rulePackages are the packages rule i uses.
func (rs *Runtimes) rulePackages(i int) []string {
	s := rs.Rules[i].Runtime
	res := &ResolvedRuntime{Plugins: map[string]string{}, Env: map[string]string{}}
	if s.Core != nil {
		res.Core = *s.Core
	}
	for k, v := range s.Plugins {
		if v != nil {
			res.Plugins[k] = *v
		}
	}
	if s.Preload != nil {
		res.Preload = *s.Preload
	}
	for k, v := range s.Env {
		if v != nil {
			res.Env[k] = *v
		}
	}
	if s.EPs != nil {
		res.EPs = *s.EPs
	}
	return res.Packages()
}

// Available returns rs without the rules that use a package have does not
// report, so a build that bundles only some packages uses only the rules
// it can satisfy.
func (rs *Runtimes) Available(have func(pkg string) bool) *Runtimes {
	out := &Runtimes{Packages: rs.Packages}
	for i, r := range rs.Rules {
		ok := true
		for _, p := range rs.rulePackages(i) {
			if !have(p) {
				ok = false
				break
			}
		}
		if ok {
			out.Rules = append(out.Rules, r)
		}
	}
	return out
}

// Candidates are the packages of all rules that may match a device with
// facts, any model: rules whose match keys facts gives must match them,
// keys facts does not give may match anything. They are what a build for
// such devices bundles, without those marked Download unless downloads.
func (rs *Runtimes) Candidates(facts Facts, downloads bool) []string {
	set := map[string]bool{}
	for i, r := range rs.Rules {
		ok := true
		for k, ps := range r.Match {
			v, given := facts[k]
			if !given {
				continue
			}
			hit := false
			for _, p := range ps {
				if m, _ := path.Match(p, v); m {
					hit = true
					break
				}
			}
			if !hit {
				ok = false
				break
			}
		}
		if ok {
			for _, p := range rs.rulePackages(i) {
				if downloads || !rs.Packages[p].Download {
					set[p] = true
				}
			}
		}
	}
	out := make([]string, 0, len(set))
	for p := range set {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// RuntimesDir is where moxgo installs the packages it downloads:
// ~/.moxgo/runtimes.
func RuntimesDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".moxgo", "runtimes"), nil
}

func writeFileAtomic(dst string, data []byte) error {
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}
