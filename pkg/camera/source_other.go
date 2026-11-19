//go:build !(linux && (amd64 || arm64))

package camera

import "fmt"

func openPlatformSource(index int, cfg SourceConfig) (VideoSource, error) {
	cam, err := Open(index, Options{Width: cfg.Width, Height: cfg.Height})
	if err != nil {
		return nil, err
	}
	return newImageSource(cam, fmt.Sprintf("camera %d", index)), nil
}
