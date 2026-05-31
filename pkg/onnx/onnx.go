// Package onnx provides a zero-copy Go binding for ONNX Runtime loaded dynamically at runtime.
//
// Architecture:
//   - purego loads the .so and resolves OrtGetApiBase (cross-platform dlopen)
//   - A tiny C trampoline (vcall.c) calls ORT vtable function pointers with
//     fixed args — avoids purego.SyscallN's variadic heap allocation on the hot path
//   - All business logic (session management, tensor binding, output reading) is in Go
//   - Tensor data is passed by raw pointer (zero-copy); runtime.Pinner prevents GC interference
package onnx

/*
#include <stdint.h>

extern uintptr_t ort_vcall(uintptr_t fn,
    uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3,
    uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7);

extern void ort_vcall_void(uintptr_t fn, uintptr_t a0);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	ErrLibraryNotFound       = errors.New("onnx: runtime library not found")
	ErrSymbolNotFound        = errors.New("onnx: OrtGetApiBase symbol not found")
	ErrAPIVersionUnsupported = errors.New("onnx: no compatible API version")
	ErrOrt                   = errors.New("onnx: runtime error")
	ErrInputNotFound         = errors.New("onnx: input not found")
	ErrOutputNotFound        = errors.New("onnx: output not found")
	ErrDtypeMismatch         = errors.New("onnx: element type mismatch")
)

const MinAPIVersion = 11

type ElemType int32

const (
	ElemTypeUndef   ElemType = 0
	ElemTypeFloat32 ElemType = 1
	ElemTypeUint8   ElemType = 2
	ElemTypeInt8    ElemType = 3
	ElemTypeUint16  ElemType = 4
	ElemTypeInt16   ElemType = 5
	ElemTypeInt32   ElemType = 6
	ElemTypeInt64   ElemType = 7
	ElemTypeString  ElemType = 8
	ElemTypeBool    ElemType = 9
	ElemTypeFloat16 ElemType = 10
	ElemTypeFloat64 ElemType = 11
	ElemTypeUint32  ElemType = 12
	ElemTypeUint64  ElemType = 13
)

func (e ElemType) String() string {
	switch e {
	case ElemTypeFloat32:
		return "float32"
	case ElemTypeFloat64:
		return "float64"
	case ElemTypeInt8:
		return "int8"
	case ElemTypeUint8:
		return "uint8"
	case ElemTypeInt16:
		return "int16"
	case ElemTypeUint16:
		return "uint16"
	case ElemTypeInt32:
		return "int32"
	case ElemTypeInt64:
		return "int64"
	case ElemTypeUint32:
		return "uint32"
	case ElemTypeUint64:
		return "uint64"
	case ElemTypeBool:
		return "bool"
	case ElemTypeFloat16:
		return "float16"
	case ElemTypeString:
		return "string"
	default:
		return fmt.Sprintf("unknown(%d)", int(e))
	}
}

func (e ElemType) Size() int {
	switch e {
	case ElemTypeFloat32, ElemTypeInt32, ElemTypeUint32:
		return 4
	case ElemTypeFloat64, ElemTypeInt64, ElemTypeUint64:
		return 8
	case ElemTypeFloat16, ElemTypeInt16, ElemTypeUint16:
		return 2
	case ElemTypeInt8, ElemTypeUint8, ElemTypeBool:
		return 1
	default:
		return 0
	}
}

type TensorInfo struct {
	Name  string
	Dtype ElemType
	Shape []int64
}

// ORT API vtable slot indices.
const (
	slotGetErrorMessage                       = 2
	slotCreateEnv                             = 3
	slotCreateSession                         = 7
	slotRun                                   = 9
	slotCreateSessionOptions                  = 10
	slotSetSessionGraphOptimizationLevel      = 23
	slotSetIntraOpNumThreads                  = 24
	slotSetInterOpNumThreads                  = 25
	slotSessionGetInputCount                  = 30
	slotSessionGetOutputCount                 = 31
	slotSessionGetInputTypeInfo               = 33
	slotSessionGetOutputTypeInfo              = 34
	slotSessionGetInputName                   = 36
	slotSessionGetOutputName                  = 37
	slotCreateTensorWithDataAsOrtValue        = 49
	slotGetTensorMutableData                  = 51
	slotCastTypeInfoToTensorInfo              = 55
	slotGetTensorElementType                  = 60
	slotGetDimensionsCount                    = 61
	slotGetDimensions                         = 62
	slotGetTensorTypeAndShape                 = 65
	slotCreateCpuMemoryInfo                   = 69
	slotAllocatorFree                         = 76
	slotGetAllocatorWithDefaultOptions        = 78
	slotReleaseEnv                            = 92
	slotReleaseStatus                         = 93
	slotReleaseMemoryInfo                     = 94
	slotReleaseSession                        = 95
	slotReleaseValue                          = 96
	slotReleaseTypeInfo                       = 98
	slotReleaseTensorTypeAndShapeInfo         = 99
	slotReleaseSessionOptions                 = 100
	slotSessionGetModelMetadata               = 111
	slotModelMetadataGetProducerName          = 112
	slotModelMetadataGetGraphName             = 113
	slotModelMetadataGetDomain                = 114
	slotModelMetadataGetDescription           = 115
	slotModelMetadataLookupCustomMetadataMap  = 116
	slotModelMetadataGetVersion               = 117
	slotReleaseModelMetadata                  = 118
	slotModelMetadataGetCustomMetadataMapKeys = 123
	slotGetAvailableProviders                 = 125
	slotReleaseAvailableProviders             = 126
	slotSessionOptionsAppendExecutionProvider = 216
)

// Runtime holds the dynamically loaded ONNX Runtime library and API pointers.
type Runtime struct {
	lib        uintptr
	api        uintptr
	APIVersion uint32
	lastError  string

	// Cached from OrtApiBase for VersionString
	versionStr string
}

// vtable reads a function pointer from the OrtApi vtable at the given slot.
func (rt *Runtime) vtable(slot int) C.uintptr_t { //nolint:govet
	ptr := *(*uintptr)(unsafe.Pointer(rt.api + uintptr(slot)*unsafe.Sizeof(uintptr(0)))) //nolint:govet
	return C.uintptr_t(ptr)
}

// call invokes ORT API slot with args, checks OrtStatus* return.
func (rt *Runtime) call(slot int, a0, a1, a2, a3, a4, a5, a6, a7 uintptr) error {
	status := C.ort_vcall(rt.vtable(slot),
		C.uintptr_t(a0), C.uintptr_t(a1), C.uintptr_t(a2), C.uintptr_t(a3),
		C.uintptr_t(a4), C.uintptr_t(a5), C.uintptr_t(a6), C.uintptr_t(a7))
	return rt.check(uintptr(status))
}

// release calls a void Release function.
func (rt *Runtime) release(slot int, ptr uintptr) {
	if ptr == 0 {
		return
	}
	C.ort_vcall_void(rt.vtable(slot), C.uintptr_t(ptr))
}

// check converts an OrtStatus* into a Go error.
func (rt *Runtime) check(status uintptr) error {
	if status == 0 {
		rt.lastError = ""
		return nil
	}
	// GetErrorMessage(status) -> const char*
	msgPtr := C.ort_vcall(rt.vtable(slotGetErrorMessage),
		C.uintptr_t(status), 0, 0, 0, 0, 0, 0, 0)
	rt.lastError = goString(uintptr(msgPtr))
	// ReleaseStatus(status)
	C.ort_vcall_void(rt.vtable(slotReleaseStatus), C.uintptr_t(status))
	return fmt.Errorf("%w: %s", ErrOrt, rt.lastError)
}

var defaultCandidates = []string{
	"./libonnxruntime.so",
	"./libonnxruntime.so.1",
	"libonnxruntime.so.1",
	"libonnxruntime.so",
	"/usr/lib/libonnxruntime.so",
	"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
	"/usr/local/lib/libonnxruntime.so",
}

var darwinCandidates = []string{
	"./libonnxruntime.dylib",
	"libonnxruntime.dylib",
	"/usr/local/lib/libonnxruntime.dylib",
	"/opt/homebrew/lib/libonnxruntime.dylib",
}

var windowsCandidates = []string{
	".\\onnxruntime.dll",
	"onnxruntime.dll",
}

// Load attempts to open ONNX Runtime from platform-default paths.
func Load() (*Runtime, error) {
	candidates := defaultCandidates
	switch runtime.GOOS {
	case "darwin":
		candidates = darwinCandidates
	case "windows":
		candidates = windowsCandidates
	}
	var lastErr error
	for _, path := range candidates {
		rt, err := LoadFrom(path)
		if err == nil {
			return rt, nil
		}
		lastErr = err
	}
	if lastErr != nil {
		return nil, lastErr
	}
	return nil, ErrLibraryNotFound
}

// LoadFrom opens ONNX Runtime from an explicit shared library path.
func LoadFrom(path string) (*Runtime, error) {
	lib, err := dlOpen(path)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrLibraryNotFound, err)
	}

	sym, err := dlSym(lib, "OrtGetApiBase")
	if err != nil {
		_ = dlClose(lib)
		return nil, ErrSymbolNotFound
	}

	// Call OrtGetApiBase() via SyscallN to avoid purego.RegisterFunc's Pinner leak.
	apiBase, _, _ := purego.SyscallN(sym)
	if apiBase == 0 {
		_ = dlClose(lib)
		return nil, ErrAPIVersionUnsupported
	}

	// OrtApiBase layout: [GetApi, GetVersionString]
	getApiPtr := *(*uintptr)(unsafe.Pointer(apiBase))                                 //nolint:govet
	getVersionPtr := *(*uintptr)(unsafe.Pointer(apiBase + unsafe.Sizeof(uintptr(0)))) //nolint:govet

	// GetVersionString()
	verStrPtr, _, _ := purego.SyscallN(getVersionPtr)
	versionStr := goString(verStrPtr)
	runtimeVer := parseMinorVersion(versionStr)

	// Negotiate API version: GetApi(version) -> *OrtApi
	startVer := runtimeVer
	if startVer == 0 {
		startVer = 26
	}
	var api uintptr
	var usedVer uint32
	for v := startVer; v >= MinAPIVersion; v-- {
		a, _, _ := purego.SyscallN(getApiPtr, uintptr(v))
		if a != 0 {
			api = a
			usedVer = v
			break
		}
	}
	if api == 0 {
		_ = dlClose(lib)
		return nil, ErrAPIVersionUnsupported
	}

	return &Runtime{
		lib:        lib,
		api:        api,
		APIVersion: usedVer,
		versionStr: versionStr,
	}, nil
}

func parseMinorVersion(s string) uint32 {
	var major, minor int
	_, _ = fmt.Sscanf(s, "%d.%d", &major, &minor)
	return uint32(minor)
}

// VersionString returns the ORT version (e.g. "1.18.0").
func (rt *Runtime) VersionString() string {
	return rt.versionStr
}

// Close releases the shared library handle.
func (rt *Runtime) Close() {
	if rt.lib != 0 {
		_ = dlClose(rt.lib)
		rt.lib = 0
	}
}

// LastError returns the last ORT error message.
func (rt *Runtime) LastError() string {
	return rt.lastError
}

// GetAvailableProviders returns the list of available execution providers.
func (rt *Runtime) GetAvailableProviders() ([]string, error) {
	var ptrs uintptr
	var length int32

	if err := rt.call(slotGetAvailableProviders,
		uintptr(unsafe.Pointer(&ptrs)), uintptr(unsafe.Pointer(&length)),
		0, 0, 0, 0, 0, 0); err != nil {
		return nil, err
	}
	defer rt.call(slotReleaseAvailableProviders, ptrs, uintptr(length), 0, 0, 0, 0, 0, 0) //nolint:errcheck

	result := make([]string, 0, int(length))
	for i := 0; i < int(length); i++ {
		strPtr := *(*uintptr)(unsafe.Pointer(ptrs + uintptr(i)*unsafe.Sizeof(uintptr(0)))) //nolint:govet
		result = append(result, goString(strPtr))
	}
	return result, nil
}

// goString reads a null-terminated C string from a pointer.
//
//nolint:govet
func goString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	var length int
	for {
		b := *(*byte)(unsafe.Pointer(ptr + uintptr(length)))
		if b == 0 {
			break
		}
		length++
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(ptr)), length))
}

// cstring allocates a null-terminated byte slice from a Go string.
func cstring(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}
