package onnx

// TensorElem is the constraint for types usable as tensor elements.
type TensorElem interface {
	~float32 | ~float64 | ~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~int64 | ~uint32 | ~uint64
}

func elemTypeOf[T TensorElem]() ElemType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return ElemTypeFloat32
	case float64:
		return ElemTypeFloat64
	case int8:
		return ElemTypeInt8
	case uint8:
		return ElemTypeUint8
	case int16:
		return ElemTypeInt16
	case uint16:
		return ElemTypeUint16
	case int32:
		return ElemTypeInt32
	case int64:
		return ElemTypeInt64
	case uint32:
		return ElemTypeUint32
	case uint64:
		return ElemTypeUint64
	default:
		return ElemTypeUndef
	}
}
