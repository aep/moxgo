//go:build !cgo

package onnx

import "github.com/ebitengine/purego"

// ZeroAllocCalls reports whether calls into ONNX Runtime allocate nothing
// on the Go heap: true with cgo, false for the purego path of builds
// without cgo, which allocates a few small objects per call.
const ZeroAllocCalls = false

// ortCall calls a C function pointer with up to 8 pointer-sized args. All
// ORT API functions take pointer/size_t args and return OrtStatus*; extra
// args beyond what the callee reads are harmless.
//
// It goes through purego rather than cgo so that moxgo, and whatever embeds
// it, builds with CGO_ENABLED=0. uintptrescapes keeps the objects that
// uintptr(unsafe.Pointer(&x)) arguments point to alive and in place for
// the call, as cgo did.
//
//go:uintptrescapes
func ortCall(fn, a0, a1, a2, a3, a4, a5, a6, a7 uintptr) uintptr {
	r, _, _ := purego.SyscallN(fn, a0, a1, a2, a3, a4, a5, a6, a7)
	return r
}

// ortCallVoid is for Release functions that return void.
//
//go:uintptrescapes
func ortCallVoid(fn, a0 uintptr) {
	purego.SyscallN(fn, a0)
}
