package camera

import (
	"errors"
	"image"
	"strings"
)

var (
	ErrNotSupported = errors.New("camera: not supported on this platform")
	// ErrTimeout says no frame arrived in time. The source can still recover.
	ErrTimeout = errors.New("camera: timed out")
	errAgain   = errors.New("camera: no frame ready")
)

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
	// FPS is the requested frame rate. 0 leaves the driver's default.
	FPS int
	// Format forces a V4L2 pixel format (a fourcc). 0 picks the best one.
	Format uint32
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
// as MJPEG streams and rtsp:// natively through gaal, falling back to ffmpeg
// if gaal has no decoder for the stream. Other protocols, rtsps:// among
// them, require ffmpeg in PATH.
func OpenStream(url string) (Camera, error) {
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return openMJPEGStream(url)
	}
	if isRTSP(url) {
		src, err := OpenVideoStream(url, SourceConfig{})
		if err == nil {
			return newSourceCamera(src), nil
		}
		if !isNoDecoder(err) {
			return nil, err
		}
	}
	return openFFmpegStream(url)
}
