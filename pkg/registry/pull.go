package registry

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const DefaultRegistry = "https://registry.moxgo.ai"

func FetchIndex(registryURL string) (*Index, error) {
	resp, err := http.Get(registryURL + "/v1/index.json")
	if err != nil {
		return nil, fmt.Errorf("fetch index: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("fetch index: HTTP %d", resp.StatusCode)
	}

	var idx Index
	if err := json.NewDecoder(resp.Body).Decode(&idx); err != nil {
		return nil, fmt.Errorf("parse index: %w", err)
	}
	return &idx, nil
}

type Progress struct {
	File       string
	Total      int64
	Downloaded int64
	Done       bool
}

func Pull(registryURL, name string, progress func(Progress)) (*Manifest, error) {
	resp, err := http.Get(registryURL + "/v1/" + name + ".json")
	if err != nil {
		return nil, fmt.Errorf("fetch manifest: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == 404 {
		return nil, fmt.Errorf("model %q not found", name)
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("fetch manifest: HTTP %d", resp.StatusCode)
	}

	var m Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}

	dir, err := ModelDir(name)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	for _, f := range m.Files {
		dst := filepath.Join(dir, f.Name)

		if existing, err := hashFile(dst); err == nil && existing == f.SHA256 {
			if progress != nil {
				progress(Progress{File: f.Name, Total: f.Size, Downloaded: f.Size, Done: true})
			}
			continue
		}

		var cb func(int64)
		if progress != nil {
			cb = func(downloaded int64) {
				progress(Progress{File: f.Name, Total: f.Size, Downloaded: downloaded})
			}
		}

		blobURL := registryURL + "/v1/blobs/sha256-" + f.SHA256
		if err := downloadFile(blobURL, dst, f.SHA256, cb); err != nil {
			return nil, fmt.Errorf("download %s: %w", f.Name, err)
		}
		if progress != nil {
			progress(Progress{File: f.Name, Total: f.Size, Downloaded: f.Size, Done: true})
		}
	}

	manifestData, _ := json.MarshalIndent(&m, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), manifestData, 0644); err != nil {
		return nil, err
	}

	return &m, nil
}

func downloadFile(url, dst, expectedSHA256 string, progress func(int64)) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	tmp := dst + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}

	h := sha256.New()
	w := io.MultiWriter(f, h)

	var src io.Reader = resp.Body
	if progress != nil {
		src = &progressReader{r: resp.Body, cb: progress}
	}

	if _, err := io.Copy(w, src); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}

	got := hex.EncodeToString(h.Sum(nil))
	if got != expectedSHA256 {
		_ = os.Remove(tmp)
		return fmt.Errorf("sha256 mismatch: got %s, want %s", got, expectedSHA256)
	}

	return os.Rename(tmp, dst)
}

type progressReader struct {
	r    io.Reader
	cb   func(int64)
	n    int64
	last int64
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.r.Read(p)
	pr.n += int64(n)
	if pr.n-pr.last >= 256*1024 || err != nil {
		pr.cb(pr.n)
		pr.last = pr.n
	}
	return n, err
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
