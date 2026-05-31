package onnx

import (
	"fmt"
	"syscall"
)

func dlOpen(path string) (uintptr, error) {
	h, err := syscall.LoadLibrary(path)
	if err != nil {
		return 0, fmt.Errorf("LoadLibrary %s: %w", path, err)
	}
	return uintptr(h), nil
}

func dlSym(lib uintptr, name string) (uintptr, error) {
	proc, err := syscall.GetProcAddress(syscall.Handle(lib), name)
	if err != nil {
		return 0, fmt.Errorf("GetProcAddress %s: %w", name, err)
	}
	return proc, nil
}

func dlClose(lib uintptr) error {
	return syscall.FreeLibrary(syscall.Handle(lib))
}
