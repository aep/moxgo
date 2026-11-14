//go:build !windows

package onnx

import "github.com/ebitengine/purego"

func dlOpen(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_LAZY|purego.RTLD_LOCAL)
}

func dlOpenGlobal(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

func dlSym(lib uintptr, name string) (uintptr, error) {
	return purego.Dlsym(lib, name)
}

func dlClose(lib uintptr) error {
	return purego.Dlclose(lib)
}

// dlPreloadGlobal opens the first loadable library from names with
// RTLD_GLOBAL so its symbols can satisfy later-loaded libraries. Failures are
// ignored; the handle is intentionally never closed.
func dlPreloadGlobal(names ...string) {
	for _, n := range names {
		if _, err := purego.Dlopen(n, purego.RTLD_NOW|purego.RTLD_GLOBAL); err == nil {
			return
		}
	}
}
