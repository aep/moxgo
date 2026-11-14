package camera

import (
	"bufio"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
)

type mjpegStream struct {
	resp   *http.Response
	frames chan image.Image
	done   chan struct{}
	err    error
}

func openMJPEGStream(url string) (*mjpegStream, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("stream: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("stream: HTTP %d", resp.StatusCode)
	}

	ct := resp.Header.Get("Content-Type")
	mediaType, params, _ := mime.ParseMediaType(ct)

	s := &mjpegStream{
		resp:   resp,
		frames: make(chan image.Image, 1),
		done:   make(chan struct{}),
	}

	switch mediaType {
	case "multipart/x-mixed-replace":
		boundary := params["boundary"]
		if boundary == "" {
			resp.Body.Close()
			return nil, fmt.Errorf("stream: multipart response has no boundary")
		}
		go s.readMultipart(boundary)
	case "image/jpeg":
		// Single-shot JPEG endpoint — not a stream, but still useful.
		resp.Body.Close()
		return nil, fmt.Errorf("stream: URL returns a single JPEG, not a stream (use --file instead)")
	default:
		resp.Body.Close()
		return nil, fmt.Errorf("stream: unsupported Content-Type %q (expected multipart/x-mixed-replace)", ct)
	}

	return s, nil
}

func (s *mjpegStream) readMultipart(boundary string) {
	defer close(s.done)
	mr := multipart.NewReader(s.resp.Body, boundary)
	for {
		part, err := mr.NextPart()
		if err != nil {
			s.err = err
			return
		}
		img, err := jpeg.Decode(bufio.NewReaderSize(part, 256*1024))
		part.Close()
		if err != nil {
			continue
		}
		select {
		case <-s.frames:
		default:
		}
		s.frames <- img
	}
}

func (s *mjpegStream) NextFrame() (image.Image, error) {
	select {
	case img := <-s.frames:
		return img, nil
	case <-s.done:
		if s.err != nil && s.err != io.EOF {
			return nil, fmt.Errorf("stream: %w", s.err)
		}
		return nil, fmt.Errorf("stream: closed")
	}
}

func (s *mjpegStream) Close() error {
	return s.resp.Body.Close()
}
