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

// Options controls how a camera is opened. Zero values mean defaults.
type Options struct {
	// Width and Height are the requested capture size. The driver may pick
	// the closest size it supports. Default 640x480.
	Width, Height int
}

func (o Options) withDefaults() Options {
	if o.Width <= 0 || o.Height <= 0 {
		o.Width, o.Height = 640, 480
	}
	return o
}

func Open(index int, opts Options) (Camera, error) {
	return open(index, opts.withDefaults())
}

// OpenStream opens a video stream URL. HTTP/HTTPS URLs are handled natively
// as MJPEG streams. RTSP and other protocols require ffmpeg in PATH.
func OpenStream(url string) (Camera, error) {
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return openMJPEGStream(url)
	}
	return openFFmpegStream(url)
}
