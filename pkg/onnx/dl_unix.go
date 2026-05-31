//go:build !windows

package onnx

import "github.com/ebitengine/purego"

func dlOpen(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_LAZY|purego.RTLD_LOCAL)
}

func dlSym(lib uintptr, name string) (uintptr, error) {
	return purego.Dlsym(lib, name)
}

func dlClose(lib uintptr) error {
	return purego.Dlclose(lib)
}
