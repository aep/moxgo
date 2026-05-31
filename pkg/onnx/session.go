package onnx

/*
#include <stdint.h>
extern uintptr_t ort_vcall(uintptr_t fn,
    uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3,
    uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7);
*/
import "C"

import (
	"fmt"
	"runtime"
	"strconv"
	"unsafe"
)

const (
	OrtLoggingLevelWarning = 2
	OrtLoggingLevelError   = 3

	ortArenaAllocator = 0
	ortMemTypeDefault = 0
	ortEnableAll      = 99
)

// SessionOptions configures session creation.
type SessionOptions struct {
	ExecutionProviders []ExecutionProvider
	IntraOpThreads     int
	InterOpThreads     int
	LogLevel           int
}

// ExecutionProvider specifies a preferred EP.
type ExecutionProvider struct {
	Name    string
	Options map[string]string
}

// EPError records a failed execution provider attempt.
type EPError struct {
	Name string
	Err  error
}

// Session holds a loaded ONNX model. Must be closed with Close().
type Session struct {
	rt        *Runtime
	env       uintptr
	opts      uintptr
	sess      uintptr
	memInfo   uintptr
	allocator uintptr
	Inputs    []TensorInfo
	Outputs   []TensorInfo
	ActiveEP  string
	EPErrors  []EPError

	// Pre-resolved hot-path function pointers (read once from vtable).
	fnRun                       C.uintptr_t
	fnReleaseValue              C.uintptr_t
	fnGetTensorTypeAndShape     C.uintptr_t
	fnReleaseTensorTypeAndShape C.uintptr_t
	fnGetTensorElementType      C.uintptr_t
	fnGetDimensionsCount        C.uintptr_t
	fnGetDimensions             C.uintptr_t
	fnGetTensorMutableData      C.uintptr_t
}

// OpenSession loads a model from path with default options.
func (rt *Runtime) OpenSession(modelPath string) (*Session, error) {
	return rt.OpenSessionWith(modelPath, SessionOptions{IntraOpThreads: 1})
}

// OpenSessionWith loads a model with the given options.
func (rt *Runtime) OpenSessionWith(modelPath string, opts SessionOptions) (*Session, error) {
	s := &Session{rt: rt, ActiveEP: "CPU"}

	logLevel := opts.LogLevel
	if logLevel == 0 {
		logLevel = OrtLoggingLevelError
	}
	logid := cstring("gomax")
	if err := rt.call(slotCreateEnv,
		uintptr(logLevel),
		uintptr(unsafe.Pointer(&logid[0])),
		uintptr(unsafe.Pointer(&s.env)),
		0, 0, 0, 0, 0); err != nil {
		return nil, err
	}
	runtime.KeepAlive(logid)

	// Try each EP: append it to fresh session options, then CreateSession.
	// If CreateSession fails (EP can't handle the model), record the error and retry next EP.
	for _, ep := range opts.ExecutionProviders {
		if err := s.tryCreateSession(modelPath, opts, &ep); err != nil {
			s.EPErrors = append(s.EPErrors, EPError{Name: ep.Name, Err: err})
			continue
		}
		s.ActiveEP = ep.Name
		break
	}

	// CPU fallback if no EP succeeded or none were requested.
	if s.sess == 0 {
		if err := s.tryCreateSession(modelPath, opts, nil); err != nil {
			s.Close()
			return nil, err
		}
	}

	if err := rt.call(slotGetAllocatorWithDefaultOptions,
		uintptr(unsafe.Pointer(&s.allocator)),
		0, 0, 0, 0, 0, 0, 0); err != nil {
		s.Close()
		return nil, err
	}

	if err := rt.call(slotCreateCpuMemoryInfo,
		uintptr(ortArenaAllocator), uintptr(ortMemTypeDefault),
		uintptr(unsafe.Pointer(&s.memInfo)),
		0, 0, 0, 0, 0); err != nil {
		s.Close()
		return nil, err
	}

	var err error
	s.Inputs, err = s.collectIO(true)
	if err != nil {
		s.Close()
		return nil, err
	}
	s.Outputs, err = s.collectIO(false)
	if err != nil {
		s.Close()
		return nil, err
	}

	// Cache hot-path vtable entries.
	s.fnRun = rt.vtable(slotRun)
	s.fnReleaseValue = rt.vtable(slotReleaseValue)
	s.fnGetTensorTypeAndShape = rt.vtable(slotGetTensorTypeAndShape)
	s.fnReleaseTensorTypeAndShape = rt.vtable(slotReleaseTensorTypeAndShapeInfo)
	s.fnGetTensorElementType = rt.vtable(slotGetTensorElementType)
	s.fnGetDimensionsCount = rt.vtable(slotGetDimensionsCount)
	s.fnGetDimensions = rt.vtable(slotGetDimensions)
	s.fnGetTensorMutableData = rt.vtable(slotGetTensorMutableData)

	return s, nil
}

// tryCreateSession creates session options, optionally appends one EP, and calls CreateSession.
// On failure the session options and ORT session are cleaned up so the caller can retry.
func (s *Session) tryCreateSession(modelPath string, opts SessionOptions, ep *ExecutionProvider) error {
	rt := s.rt

	var sopts uintptr
	if err := rt.call(slotCreateSessionOptions,
		uintptr(unsafe.Pointer(&sopts)),
		0, 0, 0, 0, 0, 0, 0); err != nil {
		return err
	}

	_ = rt.call(slotSetSessionGraphOptimizationLevel, sopts, uintptr(ortEnableAll), 0, 0, 0, 0, 0, 0)
	if opts.IntraOpThreads > 0 {
		_ = rt.call(slotSetIntraOpNumThreads, sopts, uintptr(opts.IntraOpThreads), 0, 0, 0, 0, 0, 0)
	}
	if opts.InterOpThreads > 0 {
		_ = rt.call(slotSetInterOpNumThreads, sopts, uintptr(opts.InterOpThreads), 0, 0, 0, 0, 0, 0)
	}

	old := s.opts
	s.opts = sopts

	if ep != nil {
		if err := s.appendEP(*ep); err != nil {
			s.opts = old
			rt.release(slotReleaseSessionOptions, sopts)
			return err
		}
	}

	pathC := cstring(modelPath)
	var sess uintptr
	err := rt.call(slotCreateSession,
		s.env,
		uintptr(unsafe.Pointer(&pathC[0])),
		sopts,
		uintptr(unsafe.Pointer(&sess)),
		0, 0, 0, 0)
	runtime.KeepAlive(pathC)

	if err != nil {
		s.opts = old
		rt.release(slotReleaseSessionOptions, sopts)
		return err
	}

	// Success — release any previous session options and install the new ones.
	if old != 0 {
		rt.release(slotReleaseSessionOptions, old)
	}
	s.sess = sess
	return nil
}

// Provider-specific standalone functions not available in the generic V2 API.
var epStandaloneFuncs = map[string]string{
	"CUDA":                      "OrtSessionOptionsAppendExecutionProvider_CUDA",
	"CUDAExecutionProvider":     "OrtSessionOptionsAppendExecutionProvider_CUDA",
	"TensorRT":                  "OrtSessionOptionsAppendExecutionProvider_TensorRT",
	"TensorRTExecutionProvider": "OrtSessionOptionsAppendExecutionProvider_TensorRT",
	"Nnapi":                     "OrtSessionOptionsAppendExecutionProvider_Nnapi",
	"NnapiExecutionProvider":    "OrtSessionOptionsAppendExecutionProvider_Nnapi",
}

// V2 API names differ from GetAvailableProviders names.
var epV2Names = map[string]string{
	"Xnnpack": "XNNPACK",
	"WebGpu":  "WebGPU",
}

func (s *Session) appendEP(ep ExecutionProvider) error {
	if sym, ok := epStandaloneFuncs[ep.Name]; ok {
		return s.appendEPStandalone(sym, ep)
	}
	if v2, ok := epV2Names[ep.Name]; ok {
		ep.Name = v2
	}
	return s.appendEPGeneric(ep)
}

// appendEPStandalone calls OrtSessionOptionsAppendExecutionProvider_CUDA(opts, device_id).
func (s *Session) appendEPStandalone(symbol string, ep ExecutionProvider) error {
	fn, err := dlSym(s.rt.lib, symbol)
	if err != nil {
		return fmt.Errorf("dlsym %s: %w", symbol, err)
	}
	deviceID := 0
	if v, ok := ep.Options["device_id"]; ok {
		deviceID, _ = strconv.Atoi(v)
	}
	status := C.ort_vcall(C.uintptr_t(fn),
		C.uintptr_t(s.opts), C.uintptr_t(deviceID), 0, 0, 0, 0, 0, 0)
	return s.rt.check(uintptr(status))
}

func (s *Session) appendEPGeneric(ep ExecutionProvider) error {
	rt := s.rt
	nameC := cstring(ep.Name)
	n := len(ep.Options)

	keys := make([]unsafe.Pointer, n)
	vals := make([]unsafe.Pointer, n)
	keyBufs := make([][]byte, n)
	valBufs := make([][]byte, n)

	i := 0
	for k, v := range ep.Options {
		keyBufs[i] = cstring(k)
		valBufs[i] = cstring(v)
		keys[i] = unsafe.Pointer(&keyBufs[i][0])
		vals[i] = unsafe.Pointer(&valBufs[i][0])
		i++
	}

	var keysPtr, valsPtr unsafe.Pointer
	if n > 0 {
		keysPtr = unsafe.Pointer(&keys[0])
		valsPtr = unsafe.Pointer(&vals[0])
	}

	err := rt.call(slotSessionOptionsAppendExecutionProvider,
		s.opts,
		uintptr(unsafe.Pointer(&nameC[0])),
		uintptr(keysPtr),
		uintptr(valsPtr),
		uintptr(n),
		0, 0, 0)
	runtime.KeepAlive(nameC)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(vals)
	runtime.KeepAlive(keyBufs)
	runtime.KeepAlive(valBufs)
	return err
}

func (s *Session) collectIO(input bool) ([]TensorInfo, error) {
	rt := s.rt
	var count uintptr

	slot := slotSessionGetOutputCount
	if input {
		slot = slotSessionGetInputCount
	}
	if err := rt.call(slot, s.sess, uintptr(unsafe.Pointer(&count)), 0, 0, 0, 0, 0, 0); err != nil {
		return nil, err
	}

	infos := make([]TensorInfo, count)
	for i := uintptr(0); i < count; i++ {
		var namePtr uintptr
		nameSlot := slotSessionGetOutputName
		if input {
			nameSlot = slotSessionGetInputName
		}
		if err := rt.call(nameSlot, s.sess, i, s.allocator, uintptr(unsafe.Pointer(&namePtr)), 0, 0, 0, 0); err != nil {
			return nil, err
		}
		infos[i].Name = goString(namePtr)
		_ = rt.call(slotAllocatorFree, s.allocator, namePtr, 0, 0, 0, 0, 0, 0)

		var typeInfo uintptr
		tiSlot := slotSessionGetOutputTypeInfo
		if input {
			tiSlot = slotSessionGetInputTypeInfo
		}
		if err := rt.call(tiSlot, s.sess, i, uintptr(unsafe.Pointer(&typeInfo)), 0, 0, 0, 0, 0); err != nil {
			return nil, err
		}

		var tensorInfo uintptr
		if err := rt.call(slotCastTypeInfoToTensorInfo, typeInfo, uintptr(unsafe.Pointer(&tensorInfo)), 0, 0, 0, 0, 0, 0); err != nil {
			rt.release(slotReleaseTypeInfo, typeInfo)
			return nil, err
		}

		var dtype int32
		_ = rt.call(slotGetTensorElementType, tensorInfo, uintptr(unsafe.Pointer(&dtype)), 0, 0, 0, 0, 0, 0)
		infos[i].Dtype = ElemType(dtype)

		var ndims uintptr
		_ = rt.call(slotGetDimensionsCount, tensorInfo, uintptr(unsafe.Pointer(&ndims)), 0, 0, 0, 0, 0, 0)

		if ndims > 0 {
			dims := make([]int64, ndims)
			_ = rt.call(slotGetDimensions, tensorInfo, uintptr(unsafe.Pointer(&dims[0])), ndims, 0, 0, 0, 0, 0)
			infos[i].Shape = dims
			runtime.KeepAlive(dims)
		}

		rt.release(slotReleaseTypeInfo, typeInfo)
	}
	return infos, nil
}

// InputIndex returns the index of a named input, or -1.
func (s *Session) InputIndex(name string) int {
	for i, info := range s.Inputs {
		if info.Name == name {
			return i
		}
	}
	return -1
}

// OutputIndex returns the index of a named output, or -1.
func (s *Session) OutputIndex(name string) int {
	for i, info := range s.Outputs {
		if info.Name == name {
			return i
		}
	}
	return -1
}

// Close releases all ORT resources.
func (s *Session) Close() {
	if s.memInfo != 0 {
		s.rt.release(slotReleaseMemoryInfo, s.memInfo)
		s.memInfo = 0
	}
	if s.sess != 0 {
		s.rt.release(slotReleaseSession, s.sess)
		s.sess = 0
	}
	if s.opts != 0 {
		s.rt.release(slotReleaseSessionOptions, s.opts)
		s.opts = 0
	}
	if s.env != 0 {
		s.rt.release(slotReleaseEnv, s.env)
		s.env = 0
	}
}

// ModelMetadata contains model metadata.
type ModelMetadata struct {
	ProducerName string
	GraphName    string
	Domain       string
	Description  string
	Version      int64
	Custom       map[string]string
}

// Metadata returns the model metadata including custom key-value pairs.
func (s *Session) Metadata() (*ModelMetadata, error) {
	rt := s.rt
	var mm uintptr
	if err := rt.call(slotSessionGetModelMetadata, s.sess, uintptr(unsafe.Pointer(&mm)), 0, 0, 0, 0, 0, 0); err != nil {
		return nil, err
	}
	defer rt.release(slotReleaseModelMetadata, mm)

	md := &ModelMetadata{Custom: make(map[string]string)}
	md.ProducerName = s.fetchMetaString(mm, slotModelMetadataGetProducerName)
	md.GraphName = s.fetchMetaString(mm, slotModelMetadataGetGraphName)
	md.Domain = s.fetchMetaString(mm, slotModelMetadataGetDomain)
	md.Description = s.fetchMetaString(mm, slotModelMetadataGetDescription)

	var version int64
	_ = rt.call(slotModelMetadataGetVersion, mm, uintptr(unsafe.Pointer(&version)), 0, 0, 0, 0, 0, 0)
	md.Version = version

	// Enumerate custom metadata keys
	var keysPtr uintptr
	var numKeys int64
	if err := rt.call(slotModelMetadataGetCustomMetadataMapKeys, mm, s.allocator,
		uintptr(unsafe.Pointer(&keysPtr)), uintptr(unsafe.Pointer(&numKeys)),
		0, 0, 0, 0); err == nil && keysPtr != 0 {
		for i := int64(0); i < numKeys; i++ {
			keyPtr := *(*uintptr)(unsafe.Pointer(keysPtr + uintptr(i)*unsafe.Sizeof(uintptr(0)))) //nolint:govet
			key := goString(keyPtr)
			_ = rt.call(slotAllocatorFree, s.allocator, keyPtr, 0, 0, 0, 0, 0, 0)

			// Lookup value for this key
			md.Custom[key] = s.fetchCustomMeta(mm, key)
		}
		_ = rt.call(slotAllocatorFree, s.allocator, keysPtr, 0, 0, 0, 0, 0, 0)
	}

	return md, nil
}

func (s *Session) fetchCustomMeta(mm uintptr, key string) string {
	rt := s.rt
	keyC := cstring(key)
	var valPtr uintptr
	status := C.ort_vcall(rt.vtable(slotModelMetadataLookupCustomMetadataMap),
		C.uintptr_t(mm), C.uintptr_t(s.allocator),
		C.uintptr_t(uintptr(unsafe.Pointer(&keyC[0]))),
		C.uintptr_t(uintptr(unsafe.Pointer(&valPtr))),
		0, 0, 0, 0)
	runtime.KeepAlive(keyC)
	if uintptr(status) != 0 {
		_ = rt.check(uintptr(status))
		return ""
	}
	if valPtr == 0 {
		return ""
	}
	result := goString(valPtr)
	_ = rt.call(slotAllocatorFree, s.allocator, valPtr, 0, 0, 0, 0, 0, 0)
	return result
}

func (s *Session) fetchMetaString(mm uintptr, slot int) string {
	rt := s.rt
	var strPtr uintptr
	status := C.ort_vcall(rt.vtable(slot),
		C.uintptr_t(mm), C.uintptr_t(s.allocator), C.uintptr_t(uintptr(unsafe.Pointer(&strPtr))),
		0, 0, 0, 0, 0)
	if uintptr(status) != 0 {
		_ = rt.check(uintptr(status))
		return ""
	}
	if strPtr == 0 {
		return ""
	}
	result := goString(strPtr)
	_ = rt.call(slotAllocatorFree, s.allocator, strPtr, 0, 0, 0, 0, 0, 0)
	return result
}
