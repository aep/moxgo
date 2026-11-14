package registry

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestSplitName(t *testing.T) {
	for _, tc := range []struct{ name, model, tag string }{
		{"yolo11n", "yolo11n", ""},
		{"yolo11n:latest", "yolo11n", ""},
		{"yolo11n:int8", "yolo11n", "int8"},
		{"clip-vit-b32.visual:fp16", "clip-vit-b32.visual", "fp16"},
	} {
		model, tag, err := SplitName(tc.name)
		if err != nil || model != tc.model || tag != tc.tag {
			t.Errorf("SplitName(%q) = %q, %q, %v; want %q, %q", tc.name, model, tag, err, tc.model, tc.tag)
		}
	}
	for _, bad := range []string{"", ":int8", "yolo11n:", "../x", "yolo11n:../x", "a/b", "yolo11n:a/b", "yolo11n:x:y", "yolo11n@int8", ".hidden"} {
		if _, _, err := SplitName(bad); err == nil {
			t.Errorf("SplitName(%q): no error", bad)
		}
	}
}

func TestPullVariant(t *testing.T) {
	model := []byte("int8 model")
	manifest, _ := json.Marshal(Manifest{
		Files:  []File{{Name: "model.onnx", SHA256: hexSHA(model), Size: int64(len(model))}},
		Inputs: nil,
	})
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		switch r.URL.Path {
		case "/v1/yolo11n/int8.json":
			_, _ = w.Write(manifest)
		case "/v1/blobs/sha256-" + hexSHA(model):
			_, _ = w.Write(model)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("HOME", t.TempDir())

	if _, err := Pull(srv.URL, "yolo11n:int8", nil); err != nil {
		t.Fatal(err)
	}
	store, _ := StoreDir()
	got, err := os.ReadFile(filepath.Join(store, "yolo11n@int8", "model.onnx"))
	if err != nil || string(got) != string(model) {
		t.Fatalf("pulled model = %q, %v", got, err)
	}
	if dir, _ := ModelDir("yolo11n:int8"); dir != filepath.Join(store, "yolo11n@int8") {
		t.Errorf("ModelDir = %s", dir)
	}
	if dir, _ := ModelDir("yolo11n:latest"); dir != filepath.Join(store, "yolo11n") {
		t.Errorf("ModelDir(latest) = %s", dir)
	}
	if _, err := ModelDir("yolo11n:../../x"); err == nil {
		t.Errorf("ModelDir accepted a path in the tag")
	}

	// The default variant is fetched from the old path.
	if _, err := Pull(srv.URL, "yolo11n", nil); err == nil {
		t.Errorf("default variant pulled from a server that only has int8")
	}
	if paths[len(paths)-1] != "/v1/yolo11n.json" {
		t.Errorf("default variant fetched %s, want /v1/yolo11n.json", paths[len(paths)-1])
	}
	if got := nameOfDir("yolo11n@int8"); got != "yolo11n:int8" {
		t.Errorf("nameOfDir = %q", got)
	}
}
