package registry

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// InstallRuntime installs packages under root, each into root/<name>,
// fetching what is missing: archive members from the package's source
// (one download per archive), other files from their URL or, with a
// registryURL, its blobs. Every file is checked against its sha256. A package installed
// before is not checked again: its directory holds a marker naming the
// package's spec, written once all its files were verified.
func InstallRuntime(rs *Runtimes, registryURL, root string, packages []string, progress func(Progress)) error {
	type member struct {
		pkg  string
		file RuntimeFile
	}
	archives := map[string][]member{} // by source URL
	sources := map[string]*Archive{}

	for _, name := range packages {
		p := rs.Packages[name]
		if p == nil {
			return fmt.Errorf("runtime package %q not in the registry", name)
		}
		dir := filepath.Join(root, name)
		marker := markerOf(p)
		if old, err := os.ReadFile(filepath.Join(dir, ".installed")); err == nil && string(old) == marker {
			continue
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		for _, f := range p.Files {
			dst := filepath.Join(dir, f.Name)
			if got, err := hashFile(dst); err == nil && got == f.SHA256 {
				continue
			}
			switch {
			case f.Path != "":
				archives[p.Source.URL] = append(archives[p.Source.URL], member{name, f})
				sources[p.Source.URL] = p.Source
			case f.URL != "":
				if err := download(f.URL, dst, f.SHA256, f.Name, f.Size, progress); err != nil {
					return fmt.Errorf("runtime package %s: %w", name, err)
				}
			default:
				if registryURL == "" {
					return fmt.Errorf("runtime package %s: file %s has no source", name, f.Name)
				}
				url := registryURL + "/v1/blobs/sha256-" + f.SHA256
				if err := download(url, dst, f.SHA256, f.Name, f.Size, progress); err != nil {
					return fmt.Errorf("runtime package %s: %w", name, err)
				}
			}
		}
	}

	for url, members := range archives {
		src := sources[url]
		want := map[string][]member{} // by member path
		for _, m := range members {
			want[m.file.Path] = append(want[m.file.Path], m)
		}
		extract := func(memberPath string, r io.Reader) error {
			for i, m := range want[memberPath] {
				dst := filepath.Join(root, m.pkg, m.file.Name)
				if i > 0 {
					// The same member in another package: copy the first.
					first := want[memberPath][0]
					in, err := os.Open(filepath.Join(root, first.pkg, first.file.Name))
					if err != nil {
						return err
					}
					err = writeVerified(dst, in, m.file.SHA256)
					_ = in.Close()
					if err != nil {
						return fmt.Errorf("%s from %s: %w", memberPath, url, err)
					}
					continue
				}
				if err := writeVerified(dst, r, m.file.SHA256); err != nil {
					return fmt.Errorf("%s from %s: %w", memberPath, url, err)
				}
			}
			delete(want, memberPath)
			return nil
		}
		if err := fetchArchive(root, src, extract, progress); err != nil {
			return err
		}
		if len(want) > 0 {
			var missing []string
			for p := range want {
				missing = append(missing, p)
			}
			return fmt.Errorf("%s has no %s", url, strings.Join(missing, ", "))
		}
	}

	for _, name := range packages {
		p := rs.Packages[name]
		if err := writeFileAtomic(filepath.Join(root, name, ".installed"), []byte(markerOf(p))); err != nil {
			return err
		}
	}
	return nil
}

// markerOf identifies a package's contents.
func markerOf(p *RuntimePackage) string {
	b, _ := json.Marshal(p.Files)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// download fetches url to dst, checking its sha256.
func download(url, dst, sha, name string, size int64, progress func(Progress)) error {
	var cb func(int64)
	if progress != nil {
		cb = func(n int64) { progress(Progress{File: name, Total: size, Downloaded: n}) }
	}
	if err := downloadFile(url, dst, sha, cb); err != nil {
		return fmt.Errorf("download %s: %w", name, err)
	}
	if progress != nil {
		progress(Progress{File: name, Total: size, Downloaded: size, Done: true})
	}
	return nil
}

// fetchArchive downloads src into a directory of its own under
// root/.downloads, checks it, hands each member to extract and deletes the
// download. The directory is this call's alone: another process fetching
// the same archive at once cannot replace or delete it.
func fetchArchive(root string, src *Archive, extract func(path string, r io.Reader) error, progress func(Progress)) error {
	downloads := filepath.Join(root, ".downloads")
	if err := os.MkdirAll(downloads, 0o755); err != nil {
		return err
	}
	tmpDir, err := os.MkdirTemp(downloads, src.SHA256+".*")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()
	tmp := filepath.Join(tmpDir, "archive")
	name := src.URL[strings.LastIndex(src.URL, "/")+1:]
	if err := download(src.URL, tmp, src.SHA256, name, src.Size, progress); err != nil {
		return err
	}

	format := src.Format
	if format == "" {
		switch {
		case strings.HasSuffix(src.URL, ".tar.gz"), strings.HasSuffix(src.URL, ".tgz"):
			format = "tar.gz"
		default:
			format = "zip"
		}
	}
	switch format {
	case "zip":
		zr, err := zip.OpenReader(tmp)
		if err != nil {
			return fmt.Errorf("%s: %w", src.URL, err)
		}
		defer func() { _ = zr.Close() }()
		for _, f := range zr.File {
			rc, err := f.Open()
			if err != nil {
				return err
			}
			err = extract(f.Name, rc)
			_ = rc.Close()
			if err != nil {
				return err
			}
		}
	case "tar.gz":
		fh, err := os.Open(tmp)
		if err != nil {
			return err
		}
		defer func() { _ = fh.Close() }()
		gz, err := gzip.NewReader(fh)
		if err != nil {
			return fmt.Errorf("%s: %w", src.URL, err)
		}
		tr := tar.NewReader(gz)
		for {
			h, err := tr.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				return fmt.Errorf("%s: %w", src.URL, err)
			}
			if h.Typeflag != tar.TypeReg {
				continue
			}
			if err := extract(strings.TrimPrefix(h.Name, "./"), tr); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("%s: unknown archive format %q", src.URL, format)
	}
	return nil
}

// writeVerified writes r to dst if its sha256 is sha.
func writeVerified(dst string, r io.Reader, sha string) error {
	f, err := createTemp(dst)
	if err != nil {
		return err
	}
	tmp := f.Name()
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(f, h), r); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != sha {
		_ = os.Remove(tmp)
		return fmt.Errorf("sha256 mismatch: got %s, want %s", got, sha)
	}
	return os.Rename(tmp, dst)
}

// BundleRuntime puts packages into out for a build to ship, fetching them
// as InstallRuntime does. A flat out gets their files only, which must
// not clash; it is filled from cache, a directory of installed packages
// kept between builds.
func BundleRuntime(rs *Runtimes, packages []string, cache string, out Layout, progress func(Progress)) error {
	if !out.Flat {
		return InstallRuntime(rs, "", out.Root, packages, progress)
	}
	if err := InstallRuntime(rs, "", cache, packages, progress); err != nil {
		return err
	}
	if err := os.MkdirAll(out.Root, 0o755); err != nil {
		return err
	}
	from := map[string]string{} // file name -> package
	for _, name := range packages {
		for _, f := range rs.Packages[name].Files {
			if other, ok := from[f.Name]; ok {
				return fmt.Errorf("packages %s and %s both have %s", other, name, f.Name)
			}
			from[f.Name] = name
			in, err := os.Open(filepath.Join(cache, name, f.Name))
			if err != nil {
				return err
			}
			err = writeVerified(filepath.Join(out.Root, f.Name), in, f.SHA256)
			_ = in.Close()
			if err != nil {
				return fmt.Errorf("%s/%s: %w", name, f.Name, err)
			}
		}
	}
	return nil
}
