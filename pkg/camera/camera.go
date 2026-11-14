package camera

import (
	"errors"
	"image"
	"strings"
)

var ErrNotSupported = errors.New("camera: not supported on this platform")

type Device struct {
	Index int
	Name  string
	Path  string
}

type Camera interface {
	NextFrame() (image.Image, error)
	Close() error
}

func List() ([]Device, error) {
	return list()
}

func Open(index int) (Camera, error) {
	return open(index)
}

// OpenStream opens a video stream URL. HTTP/HTTPS URLs are handled natively
// as MJPEG streams. RTSP and other protocols require ffmpeg in PATH.
func OpenStream(url string) (Camera, error) {
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return openMJPEGStream(url)
	}
	return openFFmpegStream(url)
}
