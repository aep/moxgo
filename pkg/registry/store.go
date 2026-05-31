package registry

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/aep/moxgo/pkg/server"
)

func StoreDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".moxgo", "models"), nil
}

func ModelDir(name string) (string, error) {
	store, err := StoreDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(store, name), nil
}

func ListLocal() ([]*server.ModelConfig, error) {
	store, err := StoreDir()
	if err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(store)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var models []*server.ModelConfig
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		cfg, err := loadLocalModel(e.Name(), filepath.Join(store, e.Name(), "manifest.json"))
		if err != nil {
			continue
		}
		models = append(models, cfg)
	}
	return models, nil
}

func loadLocalModel(name, manifestPath string) (*server.ModelConfig, error) {
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}

	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}

	dir := filepath.Dir(manifestPath)

	// Resolve label file paths relative to model directory
	for _, oc := range m.Outputs {
		if oc.Labels != "" && oc.Labels != "coco80" && oc.Labels != "coco91" {
			oc.Labels = filepath.Join(dir, oc.Labels)
		}
	}

	return &server.ModelConfig{
		Name:    name,
		Path:    filepath.Join(dir, "model.onnx"),
		EP:      m.EP,
		Threads: m.Threads,
		Inputs:  m.Inputs,
		Outputs: m.Outputs,
	}, nil
}
