package camera

import (
	"bufio"
	"fmt"
	"image"
	"io"
	"os/exec"
	"strings"
)

type ffmpegStream struct {
	cmd    *exec.Cmd
	stdout io.ReadCloser
	frames chan image.Image
	done   chan struct{}
	err    error
}

func openFFmpegStream(url string) (*ffmpegStream, error) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, fmt.Errorf("stream: ffmpeg is required for %q (not found in PATH)", url)
	}

	args := []string{"-hide_banner", "-loglevel", "error"}
	if strings.HasPrefix(url, "rtsp://") || strings.HasPrefix(url, "rtsps://") {
		args = append(args, "-rtsp_transport", "tcp")
	}
	args = append(args, "-i", url, "-f", "image2pipe", "-c:v", "ppm", "-an", "pipe:1")

	cmd := exec.Command("ffmpeg", args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stream: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("stream: ffmpeg: %w", err)
	}

	s := &ffmpegStream{
		cmd:    cmd,
		stdout: stdout,
		frames: make(chan image.Image, 1),
		done:   make(chan struct{}),
	}
	go s.readFrames()
	return s, nil
}

func (s *ffmpegStream) readFrames() {
	defer close(s.done)
	r := bufio.NewReaderSize(s.stdout, 4*1024*1024)
	for {
		img, err := readPPM(r)
		if err != nil {
			s.err = err
			return
		}
		select {
		case <-s.frames:
		default:
		}
		s.frames <- img
	}
}

func (s *ffmpegStream) NextFrame() (image.Image, error) {
	select {
	case img := <-s.frames:
		return img, nil
	case <-s.done:
		if s.err != nil && s.err != io.EOF {
			return nil, fmt.Errorf("stream: ffmpeg: %w", s.err)
		}
		return nil, fmt.Errorf("stream: closed")
	}
}

func (s *ffmpegStream) Close() error {
	if s.cmd.Process != nil {
		s.cmd.Process.Kill()
	}
	return s.cmd.Wait()
}

// readPPM reads one PPM (P6) image from r.
func readPPM(r *bufio.Reader) (*image.NRGBA, error) {
	magic, err := readPPMLine(r)
	if err != nil {
		return nil, err
	}
	if magic != "P6" {
		return nil, fmt.Errorf("expected P6, got %q", magic)
	}

	dimLine, err := readPPMLine(r)
	if err != nil {
		return nil, err
	}
	var w, h int
	if _, err := fmt.Sscanf(dimLine, "%d %d", &w, &h); err != nil {
		return nil, fmt.Errorf("bad PPM dimensions %q: %w", dimLine, err)
	}

	maxLine, err := readPPMLine(r)
	if err != nil {
		return nil, err
	}
	_ = maxLine // always 255

	data := make([]byte, w*h*3)
	if _, err := io.ReadFull(r, data); err != nil {
		return nil, err
	}

	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range w * h {
		img.Pix[i*4+0] = data[i*3+0]
		img.Pix[i*4+1] = data[i*3+1]
		img.Pix[i*4+2] = data[i*3+2]
		img.Pix[i*4+3] = 255
	}
	return img, nil
}

// readPPMLine reads a non-comment line from a PPM stream.
func readPPMLine(r *bufio.Reader) (string, error) {
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return "", err
		}
		line = strings.TrimRight(line, "\r\n")
		if len(line) > 0 && line[0] != '#' {
			return line, nil
		}
	}
}
