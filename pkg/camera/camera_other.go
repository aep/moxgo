//go:build !(linux && cgo) && !darwin

package camera

func list() ([]Device, error) {
	return nil, ErrNotSupported
}

func open(index int) (Camera, error) {
	return nil, ErrNotSupported
}
