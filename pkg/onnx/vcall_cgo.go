//go:build cgo

package onnx

/*
#include <stdint.h>

extern uintptr_t ort_vcall(uintptr_t fn,
    uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3,
    uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7);

extern void ort_vcall_void(uintptr_t fn, uintptr_t a0);
*/
import "C"

// ZeroAllocCalls reports whether calls into ONNX Runtime allocate nothing
// on the Go heap: true with cgo, false for the purego path of builds
// without cgo, which allocates a few small objects per call.
const ZeroAllocCalls = true

// ortCall calls a C function pointer with up to 8 pointer-sized args, through
// the C trampoline in vcall.c. All ORT API functions take pointer/size_t args
// and return OrtStatus*; extra args beyond what the callee reads are harmless.
func ortCall(fn, a0, a1, a2, a3, a4, a5, a6, a7 uintptr) uintptr {
	return uintptr(C.ort_vcall(C.uintptr_t(fn),
		C.uintptr_t(a0), C.uintptr_t(a1), C.uintptr_t(a2), C.uintptr_t(a3),
		C.uintptr_t(a4), C.uintptr_t(a5), C.uintptr_t(a6), C.uintptr_t(a7)))
}

// ortCallVoid is for Release functions that return void.
func ortCallVoid(fn, a0 uintptr) {
	C.ort_vcall_void(C.uintptr_t(fn), C.uintptr_t(a0))
}
