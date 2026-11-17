//go:build !(linux && cgo) && !darwin

package camera

func list() ([]Device, error) {
	return nil, ErrNotSupported
}

func open(index int, _ Options) (Camera, error) {
	return nil, ErrNotSupported
}
