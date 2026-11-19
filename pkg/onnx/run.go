package onnx

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Run holds the state for a single inference call.
// After the first Exec(), repeated Exec()+GetOutput() are zero-allocation.
type Run struct {
	sess        *Session
	inputNames  [][]byte
	inputPtrs   []unsafe.Pointer
	inputValues []uintptr
	outValues   []uintptr
	outNamesC   [][]byte
	pinner      runtime.Pinner

	// Pre-allocated for zero-alloc hot path.
	inNamePtrs  []uintptr
	outNamePtrs []uintptr
	outputs     []Output
	prepared    bool

	// Out-parameters of GetOutput. They live here, on the heap, so that it
	// stays free of allocations while ONNX Runtime writes to stable addresses.
	outInfo  uintptr
	outDtype int32
	outNDims uintptr
	outData  uintptr
}

// NewRun creates a new inference run.
func (s *Session) NewRun() *Run {
	return &Run{sess: s}
}

// Bind attaches a tensor input by name (zero-copy).
// dataPtr is used directly — caller must keep it alive until Close().
func (r *Run) Bind(name string, dataPtr unsafe.Pointer, dataLen int, dtype ElemType, shape []int64) error {
	rt := r.sess.rt
	nameC := cstring(name)

	// A rank-0 (scalar) tensor has an empty shape; pass a nil dims pointer.
	var shapePtr unsafe.Pointer
	if len(shape) > 0 {
		shapePtr = unsafe.Pointer(&shape[0])
	}

	var value uintptr
	if err := rt.call(slotCreateTensorWithDataAsOrtValue,
		r.sess.memInfo,
		uintptr(dataPtr),
		uintptr(dataLen),
		uintptr(shapePtr),
		uintptr(len(shape)),
		uintptr(dtype),
		uintptr(unsafe.Pointer(&value)),
		0); err != nil {
		return err
	}
	runtime.KeepAlive(shape)

	if value == 0 {
		return fmt.Errorf("onnx: no tensor created for input %q", name)
	}
	r.pinner.Pin(&nameC[0])
	r.inputNames = append(r.inputNames, nameC)
	r.inputPtrs = append(r.inputPtrs, unsafe.Pointer(&nameC[0]))
	r.inputValues = append(r.inputValues, value)
	r.prepared = false
	return nil
}

// BindSlice is a typed helper that binds a Go slice as input (zero-copy).
func BindSlice[T TensorElem](r *Run, name string, data []T, shape []int64) error {
	if len(data) == 0 {
		return fmt.Errorf("onnx: empty input data for %q", name)
	}
	r.pinner.Pin(&data[0])
	return r.Bind(
		name,
		unsafe.Pointer(&data[0]),
		len(data)*int(unsafe.Sizeof(data[0])),
		elemTypeOf[T](),
		shape,
	)
}

// BindBool binds a []bool as an ONNX bool tensor (zero-copy). bool is not part
// of the TensorElem numeric constraint, so it needs its own helper. A bool is
// one byte, matching ONNX's bool element type. Pass an empty shape for a scalar.
func BindBool(r *Run, name string, data []bool, shape []int64) error {
	if len(data) == 0 {
		return fmt.Errorf("onnx: empty input data for %q", name)
	}
	r.pinner.Pin(&data[0])
	return r.Bind(name, unsafe.Pointer(&data[0]), len(data), ElemTypeBool, shape)
}

// prepare builds cached pointer arrays on first call.
func (r *Run) prepare() {
	nOut := len(r.sess.Outputs)

	if len(r.outNamesC) == 0 {
		r.outNamesC = make([][]byte, nOut)
		for i, o := range r.sess.Outputs {
			r.outNamesC[i] = cstring(o.Name)
		}
	}

	r.inNamePtrs = make([]uintptr, len(r.inputPtrs))
	for i, p := range r.inputPtrs {
		r.inNamePtrs[i] = uintptr(p)
	}

	r.outNamePtrs = make([]uintptr, nOut)
	for i, n := range r.outNamesC {
		r.outNamePtrs[i] = uintptr(unsafe.Pointer(&n[0]))
	}

	r.outValues = make([]uintptr, nOut)

	r.outputs = make([]Output, nOut)
	for i, info := range r.sess.Outputs {
		r.outputs[i].Shape = make([]int64, len(info.Shape))
	}

	r.prepared = true
}

// Exec runs inference. Zero-allocation after first call.
func (r *Run) Exec() error {
	sess := r.sess

	if !r.prepared {
		r.prepare()
	}

	// Release prior output OrtValues
	for i, v := range r.outValues {
		if v != 0 {
			ortCallVoid(sess.fnReleaseValue, uintptr(v))
			r.outValues[i] = 0
		}
	}

	var inNamesPtr, inValsPtr, outNamesPtr, outValsPtr uintptr
	if len(r.inNamePtrs) > 0 {
		inNamesPtr = uintptr(unsafe.Pointer(&r.inNamePtrs[0]))
		inValsPtr = uintptr(unsafe.Pointer(&r.inputValues[0]))
	}
	nOut := len(r.outValues)
	if nOut > 0 {
		outNamesPtr = uintptr(unsafe.Pointer(&r.outNamePtrs[0]))
		outValsPtr = uintptr(unsafe.Pointer(&r.outValues[0]))
	}

	status := ortCall(sess.fnRun,
		uintptr(sess.sess),
		0, // RunOptions
		uintptr(inNamesPtr),
		uintptr(inValsPtr),
		uintptr(len(r.inputValues)),
		uintptr(outNamesPtr),
		uintptr(nOut),
		uintptr(outValsPtr),
	)

	runtime.KeepAlive(r.inNamePtrs)
	runtime.KeepAlive(r.outNamePtrs)
	runtime.KeepAlive(r.inputValues)
	runtime.KeepAlive(r.outValues)
	runtime.KeepAlive(r.outNamesC)

	return sess.rt.check(uintptr(status))
}

// Output holds a zero-copy view of an output tensor.
// Valid until next Exec() or Close().
type Output struct {
	Ptr   unsafe.Pointer
	Len   int
	Dtype ElemType
	Shape []int64
}

// GetOutput returns the i-th output. Zero-allocation (writes into pre-allocated Output).
func (r *Run) GetOutput(index int) (*Output, error) {
	if index < 0 || index >= len(r.outValues) {
		return nil, ErrOutputNotFound
	}
	v := r.outValues[index]
	if v == 0 {
		return nil, ErrOutputNotFound
	}

	sess := r.sess
	out := &r.outputs[index]

	// GetTensorTypeAndShape
	status := ortCall(sess.fnGetTensorTypeAndShape,
		uintptr(v), uintptr(unsafe.Pointer(&r.outInfo)),
		0, 0, 0, 0, 0, 0)
	if err := sess.rt.check(uintptr(status)); err != nil {
		return nil, err
	}

	info := r.outInfo

	// GetTensorElementType
	ortCall(sess.fnGetTensorElementType,
		uintptr(info), uintptr(unsafe.Pointer(&r.outDtype)),
		0, 0, 0, 0, 0, 0)
	out.Dtype = ElemType(r.outDtype)

	// GetDimensionsCount
	ortCall(sess.fnGetDimensionsCount,
		uintptr(info), uintptr(unsafe.Pointer(&r.outNDims)),
		0, 0, 0, 0, 0, 0)
	ndims := r.outNDims

	// GetDimensions — reuse pre-allocated shape buffer
	if int(ndims) > cap(out.Shape) {
		out.Shape = make([]int64, ndims)
	} else {
		out.Shape = out.Shape[:ndims]
	}
	if ndims > 0 {
		ortCall(sess.fnGetDimensions,
			uintptr(info), uintptr(uintptr(unsafe.Pointer(&out.Shape[0]))), uintptr(ndims),
			0, 0, 0, 0, 0)
	}

	ortCallVoid(sess.fnReleaseTensorTypeAndShape, uintptr(info))

	nElems := 1
	for _, d := range out.Shape {
		nElems *= int(d)
	}
	out.Len = nElems

	// GetTensorMutableData
	status = ortCall(sess.fnGetTensorMutableData,
		uintptr(v), uintptr(unsafe.Pointer(&r.outData)),
		0, 0, 0, 0, 0, 0)
	if err := sess.rt.check(uintptr(status)); err != nil {
		return nil, err
	}
	out.Ptr = unsafe.Pointer(r.outData) //nolint:govet

	return out, nil
}

// GetOutputByName returns a named output tensor.
func (r *Run) GetOutputByName(name string) (*Output, error) {
	idx := r.sess.OutputIndex(name)
	if idx < 0 {
		return nil, fmt.Errorf("%w: %q", ErrOutputNotFound, name)
	}
	return r.GetOutput(idx)
}

// OutputSlice returns the i-th output as a typed Go slice (zero-copy).
func OutputSlice[T TensorElem](r *Run, index int) ([]T, []int64, error) {
	out, err := r.GetOutput(index)
	if err != nil {
		return nil, nil, err
	}
	want := elemTypeOf[T]()
	if out.Dtype != want {
		return nil, nil, fmt.Errorf("%w: got %s, want %s", ErrDtypeMismatch, out.Dtype, want)
	}
	slice := unsafe.Slice((*T)(out.Ptr), out.Len)
	return slice, out.Shape, nil
}

// Close releases all ORT values and unpins memory.
func (r *Run) Close() {
	sess := r.sess
	for _, v := range r.inputValues {
		if v != 0 {
			ortCallVoid(sess.fnReleaseValue, uintptr(v))
		}
	}
	for _, v := range r.outValues {
		if v != 0 {
			ortCallVoid(sess.fnReleaseValue, uintptr(v))
		}
	}
	r.pinner.Unpin()
	r.inputNames = nil
	r.inputPtrs = nil
	r.inputValues = nil
	r.outValues = nil
	r.outNamesC = nil
	r.inNamePtrs = nil
	r.outNamePtrs = nil
	r.outputs = nil
	r.prepared = false
}
