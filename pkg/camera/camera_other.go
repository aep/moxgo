//go:build !(linux && (amd64 || arm64)) && !darwin

package camera

func list() ([]Device, error) {
	return nil, ErrNotSupported
}

func open(index int, _ Options) (Camera, error) {
	return nil, ErrNotSupported
}
