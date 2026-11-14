package registry

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

// The levels of the example: Android, a Qualcomm phone, a phone, and the
// phone running a model that needs int8.
const testRuntimes = `{
  "packages": {
    "ort": {"files": [{"name": "libort.so", "sha256": "x"}]},
    "ort-int8": {"files": [{"name": "libort.so", "sha256": "x"}]},
    "qnn": {"files": [{"name": "libqnn.so", "sha256": "x"}, {"name": "libQnnHtp.so", "sha256": "x"}]},
    "nnapi": {"files": [{"name": "libnn.so", "sha256": "x"}]}
  },
  "rules": [
    {"comment": "android", "match": {"os": "android"},
     "runtime": {"core": "ort/libort.so", "env": {"A": "1", "B": "2"}, "eps": [{"name": "XNNPACK"}]}},
    {"comment": "qualcomm", "match": {"os": "android", "soc.manufacturer": "QTI"},
     "runtime": {"plugins": {"QNN": "qnn/libqnn.so"},
                 "eps": [{"name": "QNN", "options": {"backend_path": "${pkg:qnn}/libQnnHtp.so"}}, {"name": "XNNPACK"}]}},
    {"comment": "this phone", "match": {"os": "android", "soc.manufacturer": "QTI", "device": ["pa1q", "pa2*"]},
     "runtime": {"env": {"B": null, "C": "3"}}},
    {"comment": "this phone, int8 models", "match": {"device": "pa1q"}, "needs": ["int8"],
     "runtime": {"core": "ort-int8/libort.so", "plugins": {"QNN": null}, "eps": []}}
  ]
}`

func mustRuntimes(t *testing.T) *Runtimes {
	t.Helper()
	rs, err := ParseRuntimes([]byte(testRuntimes))
	if err != nil {
		t.Fatal(err)
	}
	return rs
}

func TestResolveCascade(t *testing.T) {
	rs := mustRuntimes(t)
	android := Facts{"os": "android"}
	qcom := Facts{"os": "android", "soc.manufacturer": "QTI", "device": "x"}
	phone := Facts{"os": "android", "soc.manufacturer": "QTI", "device": "pa1q"}

	for _, tc := range []struct {
		name    string
		facts   Facts
		needs   []string
		core    string
		plugins map[string]string
		env     map[string]string
		eps     []string
		rules   []int
	}{
		{"linux", Facts{"os": "linux"}, nil, "", map[string]string{}, map[string]string{}, nil, nil},
		{"android", android, nil, "ort/libort.so", map[string]string{},
			map[string]string{"A": "1", "B": "2"}, []string{"XNNPACK"}, []int{0}},
		{"qualcomm", qcom, nil, "ort/libort.so", map[string]string{"QNN": "qnn/libqnn.so"},
			map[string]string{"A": "1", "B": "2"}, []string{"QNN", "XNNPACK"}, []int{0, 1}},
		{"phone", phone, nil, "ort/libort.so", map[string]string{"QNN": "qnn/libqnn.so"},
			map[string]string{"A": "1", "C": "3"}, []string{"QNN", "XNNPACK"}, []int{0, 1, 2}},
		{"phone glob", Facts{"os": "android", "soc.manufacturer": "QTI", "device": "pa2q"}, nil,
			"ort/libort.so", map[string]string{"QNN": "qnn/libqnn.so"},
			map[string]string{"A": "1", "C": "3"}, []string{"QNN", "XNNPACK"}, []int{0, 1, 2}},
		// The needs rule matches one fact only, yet applies last.
		{"phone int8", phone, []string{"int8", "fp16"}, "ort-int8/libort.so", map[string]string{},
			map[string]string{"A": "1", "C": "3"}, []string{}, []int{0, 1, 2, 3}},
		{"other phone int8", qcom, []string{"int8"}, "ort/libort.so", map[string]string{"QNN": "qnn/libqnn.so"},
			map[string]string{"A": "1", "B": "2"}, []string{"QNN", "XNNPACK"}, []int{0, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res := rs.Resolve(tc.facts, tc.needs)
			if res.Core != tc.core {
				t.Errorf("core = %q, want %q", res.Core, tc.core)
			}
			if !reflect.DeepEqual(res.Plugins, tc.plugins) {
				t.Errorf("plugins = %v, want %v", res.Plugins, tc.plugins)
			}
			if !reflect.DeepEqual(res.Env, tc.env) {
				t.Errorf("env = %v, want %v", res.Env, tc.env)
			}
			var eps []string
			if res.HasEPs {
				eps = []string{}
				for _, ep := range res.EPs {
					eps = append(eps, ep.Name)
				}
			}
			if !reflect.DeepEqual(eps, tc.eps) {
				t.Errorf("eps = %v, want %v", eps, tc.eps)
			}
			if !reflect.DeepEqual(res.Rules, tc.rules) {
				t.Errorf("rules = %v, want %v", res.Rules, tc.rules)
			}
		})
	}
}

func TestResolvedPackagesAndExpand(t *testing.T) {
	rs := mustRuntimes(t)
	res := rs.Resolve(Facts{"os": "android", "soc.manufacturer": "QTI"}, nil)
	if got, want := res.Packages(), []string{"ort", "qnn"}; !reflect.DeepEqual(got, want) {
		t.Errorf("packages = %v, want %v", got, want)
	}
	opt := res.EPs[0].Options["backend_path"]
	nested, flat := Layout{Root: "/rt"}, Layout{Root: "/lib", Flat: true}
	if got, want := ExpandIn(nested, opt), filepath.Join("/rt", "qnn", "libQnnHtp.so"); got != want {
		t.Errorf("ExpandIn = %q, want %q", got, want)
	}
	if got, want := ExpandIn(flat, opt), filepath.Join("/lib", "libQnnHtp.so"); got != want {
		t.Errorf("flat ExpandIn = %q, want %q", got, want)
	}
	if got, want := PathIn(nested, res.Core), filepath.Join("/rt", "ort", "libort.so"); got != want {
		t.Errorf("PathIn = %q, want %q", got, want)
	}
	if got, want := PathIn(flat, res.Core), filepath.Join("/lib", "libort.so"); got != want {
		t.Errorf("flat PathIn = %q, want %q", got, want)
	}
}

// A build without the QNN package uses the rules it can satisfy only.
func TestAvailable(t *testing.T) {
	rs := mustRuntimes(t).Available(func(pkg string) bool { return pkg != "qnn" })
	phone := Facts{"os": "android", "soc.manufacturer": "QTI", "device": "pa1q"}
	res := rs.Resolve(phone, nil)
	if len(res.Plugins) != 0 || len(res.EPs) != 1 || res.EPs[0].Name != "XNNPACK" {
		t.Errorf("without qnn: plugins %v eps %v, want none and XNNPACK", res.Plugins, res.EPs)
	}
	// The phone rule uses no package, so it still applies.
	if !reflect.DeepEqual(res.Env, map[string]string{"A": "1", "C": "3"}) {
		t.Errorf("without qnn: env = %v", res.Env)
	}
}

func TestCandidates(t *testing.T) {
	rs := mustRuntimes(t)
	rs.Packages["qnn"].Download = true
	if got, want := rs.Candidates(Facts{"os": "android"}, false), []string{"ort", "ort-int8"}; !reflect.DeepEqual(got, want) {
		t.Errorf("android = %v, want %v", got, want)
	}
	if got, want := rs.Candidates(Facts{"os": "android"}, true), []string{"ort", "ort-int8", "qnn"}; !reflect.DeepEqual(got, want) {
		t.Errorf("android with downloads = %v, want %v", got, want)
	}
	// The int8 rule matches a device, not an os, so it may match Linux.
	if got, want := rs.Candidates(Facts{"os": "linux"}, true), []string{"ort-int8"}; !reflect.DeepEqual(got, want) {
		t.Errorf("linux = %v, want %v", got, want)
	}
	// A fact that rules out the Qualcomm rules.
	if got, want := rs.Candidates(Facts{"os": "android", "soc.manufacturer": "Google"}, true), []string{"ort", "ort-int8"}; !reflect.DeepEqual(got, want) {
		t.Errorf("google = %v, want %v", got, want)
	}
}

func TestDefaultRuntimes(t *testing.T) {
	rs := DefaultRuntimes()
	for _, f := range []Facts{
		{"os": "linux", "arch": "amd64"}, {"os": "linux", "arch": "arm64"},
		{"os": "darwin", "arch": "arm64"}, {"os": "windows", "arch": "amd64"},
		{"os": "android", "arch": "arm64"}, {"os": "android", "arch": "arm64", "soc.manufacturer": "QTI"},
	} {
		res := rs.Resolve(f, nil)
		if res.Core == "" {
			t.Errorf("%v: no core", f)
		}
		if rs.Packages[strings.SplitN(res.Core, "/", 2)[0]].Download {
			t.Errorf("%v: core %s is downloaded, not bundled", f, res.Core)
		}
	}
	qcom := rs.Resolve(Facts{"os": "android", "arch": "arm64", "soc.manufacturer": "QTI"}, nil)
	if len(qcom.EPs) == 0 || qcom.EPs[0].Name != "QNN" || qcom.Plugins["QNN"] == "" {
		t.Errorf("qualcomm: eps %v plugins %v, want QNN first", qcom.EPs, qcom.Plugins)
	}
}

func TestParseRuntimesRejects(t *testing.T) {
	for name, js := range map[string]string{
		"unknown package": `{"packages": {}, "rules": [{"runtime": {"core": "nope/lib.so"}}]}`,
		"unknown file":    `{"packages": {"p": {"files": []}}, "rules": [{"runtime": {"core": "p/lib.so"}}]}`,
		"no slash":        `{"packages": {}, "rules": [{"runtime": {"core": "lib.so"}}]}`,
		"unknown pkg var": `{"packages": {}, "rules": [{"runtime": {"env": {"X": "${pkg:nope}"}}}]}`,
		"bad pattern":     `{"packages": {}, "rules": [{"match": {"os": "["}, "runtime": {}}]}`,
		"path traversal":  `{"packages": {"p": {"files": [{"name": "../x", "sha256": "x"}]}}, "rules": []}`,
		"no sha256":       `{"packages": {"p": {"files": [{"name": "x"}]}}, "rules": []}`,
		"path no source":  `{"packages": {"p": {"files": [{"name": "x", "sha256": "x", "path": "a/x"}]}}, "rules": []}`,
		"bad match type":  `{"packages": {}, "rules": [{"match": {"os": 1}, "runtime": {}}]}`,
	} {
		if _, err := ParseRuntimes([]byte(js)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestParseGetprop(t *testing.T) {
	out := "[ro.soc.manufacturer]: [QTI]\n[ro.soc.model]: [SM8750]\n[empty]: []\ngarbage\n[ro.product.model]: [Pixel 9a]\n"
	got := parseGetprop([]byte(out))
	want := map[string]string{"ro.soc.manufacturer": "QTI", "ro.soc.model": "SM8750", "empty": "", "ro.product.model": "Pixel 9a"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseGetprop = %v, want %v", got, want)
	}
}

func hexSHA(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func TestInstallRuntime(t *testing.T) {
	core := []byte("core library")
	plugin := []byte("plugin library")
	stub := []byte("stub")
	direct := []byte("direct file")
	blob := []byte("registry blob")

	var zbuf bytes.Buffer
	zw := zip.NewWriter(&zbuf)
	for name, b := range map[string][]byte{"jni/arm64-v8a/libcore.so": core, "jni/arm64-v8a/libstub.so": stub, "other.txt": []byte("x")} {
		w, _ := zw.Create(name)
		_, _ = w.Write(b)
	}
	_ = zw.Close()

	var tbuf bytes.Buffer
	gz := gzip.NewWriter(&tbuf)
	tw := tar.NewWriter(gz)
	_ = tw.WriteHeader(&tar.Header{Name: "./pkg/lib/libplugin.so", Mode: 0o755, Size: int64(len(plugin)), Typeflag: tar.TypeReg})
	_, _ = tw.Write(plugin)
	_ = tw.Close()
	_ = gz.Close()

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		switch r.URL.Path {
		case "/a.aar":
			_, _ = w.Write(zbuf.Bytes())
		case "/p.tar.gz":
			_, _ = w.Write(tbuf.Bytes())
		case "/direct":
			_, _ = w.Write(direct)
		case "/v1/blobs/sha256-" + hexSHA(blob):
			_, _ = w.Write(blob)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	js := fmt.Sprintf(`{
	  "packages": {
	    "core": {"source": {"url": "%[1]s/a.aar", "sha256": "%[2]s"},
	             "files": [{"name": "libcore.so", "path": "jni/arm64-v8a/libcore.so", "sha256": "%[3]s"}]},
	    "stub": {"source": {"url": "%[1]s/a.aar", "sha256": "%[2]s"},
	             "files": [{"name": "libstub.so", "path": "jni/arm64-v8a/libstub.so", "sha256": "%[4]s"},
	                       {"name": "libcore-copy.so", "path": "jni/arm64-v8a/libcore.so", "sha256": "%[3]s"}]},
	    "plugin": {"source": {"url": "%[1]s/p.tar.gz", "sha256": "%[5]s"},
	               "files": [{"name": "libplugin.so", "path": "pkg/lib/libplugin.so", "sha256": "%[6]s"}]},
	    "loose": {"files": [{"name": "direct.bin", "url": "%[1]s/direct", "sha256": "%[7]s"},
	                        {"name": "blob.bin", "sha256": "%[8]s"}]}
	  },
	  "rules": []
	}`, srv.URL, hexSHA(zbuf.Bytes()), hexSHA(core), hexSHA(stub), hexSHA(tbuf.Bytes()), hexSHA(plugin), hexSHA(direct), hexSHA(blob))
	rs, err := ParseRuntimes([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	pkgs := []string{"core", "stub", "plugin", "loose"}
	if err := InstallRuntime(rs, srv.URL, root, pkgs, nil); err != nil {
		t.Fatal(err)
	}
	for path, want := range map[string][]byte{
		"core/libcore.so": core, "stub/libstub.so": stub, "stub/libcore-copy.so": core,
		"plugin/libplugin.so": plugin, "loose/direct.bin": direct, "loose/blob.bin": blob,
	} {
		got, err := os.ReadFile(filepath.Join(root, path))
		if err != nil || !bytes.Equal(got, want) {
			t.Errorf("%s = %q, %v; want %q", path, got, err, want)
		}
	}
	// One download per archive: a.aar, p.tar.gz, direct, blob.
	if n := hits.Load(); n != 4 {
		t.Errorf("%d downloads, want 4", n)
	}
	if left, _ := os.ReadDir(filepath.Join(root, ".downloads")); len(left) != 0 {
		t.Errorf("downloads left behind: %v", left)
	}

	// Installed packages are not fetched again.
	if err := InstallRuntime(rs, srv.URL, root, pkgs, nil); err != nil {
		t.Fatal(err)
	}
	if n := hits.Load(); n != 4 {
		t.Errorf("%d downloads after reinstall, want 4", n)
	}

	// A damaged file is fetched again once the marker is gone.
	_ = os.WriteFile(filepath.Join(root, "core", "libcore.so"), []byte("bad"), 0o644)
	_ = os.Remove(filepath.Join(root, "core", ".installed"))
	if err := InstallRuntime(rs, srv.URL, root, []string{"core"}, nil); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(root, "core", "libcore.so")); !bytes.Equal(got, core) {
		t.Errorf("repaired libcore.so = %q", got)
	}
}

func TestInstallRuntimeRejectsBadHash(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("tampered"))
	}))
	defer srv.Close()
	js := fmt.Sprintf(`{"packages": {"p": {"files": [{"name": "f", "url": "%s/f", "sha256": "%s"}]}}, "rules": []}`,
		srv.URL, hexSHA([]byte("original")))
	rs, err := ParseRuntimes([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	err = InstallRuntime(rs, srv.URL, root, []string{"p"}, nil)
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("err = %v, want a sha256 mismatch", err)
	}
	if _, err := os.Stat(filepath.Join(root, "p", "f")); !os.IsNotExist(err) {
		t.Errorf("tampered file was installed")
	}
	if _, err := os.Stat(filepath.Join(root, "p", ".installed")); !os.IsNotExist(err) {
		t.Errorf("package marked installed")
	}
}

func TestBundleRuntimeFlat(t *testing.T) {
	a, b := []byte("library a"), []byte("library b")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/a":
			_, _ = w.Write(a)
		case "/b":
			_, _ = w.Write(b)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	js := fmt.Sprintf(`{"packages": {
	  "pa": {"files": [{"name": "liba.so", "url": "%[1]s/a", "sha256": "%[2]s"}]},
	  "pb": {"files": [{"name": "libb.so", "url": "%[1]s/b", "sha256": "%[3]s"}]},
	  "clash": {"files": [{"name": "liba.so", "url": "%[1]s/b", "sha256": "%[3]s"}]}
	}, "rules": []}`, srv.URL, hexSHA(a), hexSHA(b))
	rs, err := ParseRuntimes([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	cache, out := t.TempDir(), t.TempDir()
	if err := BundleRuntime(rs, []string{"pa", "pb"}, cache, Layout{Root: out, Flat: true}, nil); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(out)
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	if want := []string{"liba.so", "libb.so"}; !reflect.DeepEqual(names, want) {
		t.Errorf("flat bundle has %v, want %v", names, want)
	}
	flat := Layout{Root: out, Flat: true}
	if !flat.Has(rs, "pa") || !flat.Has(rs, "pb") || flat.Has(rs, "nope") {
		t.Errorf("Has: pa %v pb %v nope %v", flat.Has(rs, "pa"), flat.Has(rs, "pb"), flat.Has(rs, "nope"))
	}
	err = BundleRuntime(rs, []string{"pa", "clash"}, cache, Layout{Root: t.TempDir(), Flat: true}, nil)
	if err == nil || !strings.Contains(err.Error(), "both have liba.so") {
		t.Errorf("clash: err = %v", err)
	}
}
