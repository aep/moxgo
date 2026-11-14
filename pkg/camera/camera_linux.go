//go:build linux && cgo

package camera

/*
#include <stdlib.h>

int v4l2_open(const char *path);
int v4l2_close(int fd);
int v4l2_querycap(int fd, char *name, int name_len, unsigned int *caps);
int v4l2_set_format(int fd, unsigned int pixfmt, unsigned int w, unsigned int h,
                    unsigned int *actual_pixfmt, unsigned int *actual_w, unsigned int *actual_h);
int v4l2_request_buffers(int fd, unsigned int *count);
int v4l2_query_buffer(int fd, unsigned int index, unsigned int *offset, unsigned int *length);
void *v4l2_mmap(int fd, unsigned int offset, unsigned int length);
int v4l2_munmap(void *addr, unsigned int length);
int v4l2_queue_buffer(int fd, unsigned int index);
int v4l2_dequeue_buffer(int fd, unsigned int *index, unsigned int *bytesused);
int v4l2_stream_on(int fd);
int v4l2_stream_off(int fd);
int v4l2_poll(int fd, int timeout_ms);

unsigned int v4l2_cap_video_capture(void);
unsigned int v4l2_cap_streaming(void);
unsigned int v4l2_pix_fmt_mjpeg(void);
unsigned int v4l2_pix_fmt_yuyv(void);
*/
import "C"

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"path/filepath"
	"unsafe"
)

const numBuffers = 4

type mappedBuffer struct {
	ptr    unsafe.Pointer
	length C.uint
}

type v4l2Camera struct {
	fd      C.int
	format  C.uint
	width   int
	height  int
	buffers []mappedBuffer
}

func list() ([]Device, error) {
	matches, err := filepath.Glob("/dev/video*")
	if err != nil {
		return nil, err
	}
	capVideo := C.v4l2_cap_video_capture()
	var devices []Device
	for _, path := range matches {
		cpath := C.CString(path)
		fd := C.v4l2_open(cpath)
		C.free(unsafe.Pointer(cpath))
		if fd < 0 {
			continue
		}
		var name [64]C.char
		var caps C.uint
		rc := C.v4l2_querycap(fd, &name[0], 64, &caps)
		C.v4l2_close(fd)
		if rc < 0 {
			continue
		}
		if caps&capVideo == 0 {
			continue
		}
		idx := -1
		fmt.Sscanf(filepath.Base(path), "video%d", &idx)
		devices = append(devices, Device{
			Index: idx,
			Name:  C.GoString(&name[0]),
			Path:  path,
		})
	}
	return devices, nil
}

func open(index int) (Camera, error) {
	path := fmt.Sprintf("/dev/video%d", index)
	cpath := C.CString(path)
	fd := C.v4l2_open(cpath)
	C.free(unsafe.Pointer(cpath))
	if fd < 0 {
		return nil, fmt.Errorf("camera: open %s: %s", path, cError(fd))
	}

	var name [64]C.char
	var caps C.uint
	if rc := C.v4l2_querycap(fd, &name[0], 64, &caps); rc < 0 {
		C.v4l2_close(fd)
		return nil, fmt.Errorf("camera: QUERYCAP: %s", cError(rc))
	}
	if caps&C.v4l2_cap_video_capture() == 0 {
		C.v4l2_close(fd)
		return nil, fmt.Errorf("camera: %s does not support video capture", path)
	}
	if caps&C.v4l2_cap_streaming() == 0 {
		C.v4l2_close(fd)
		return nil, fmt.Errorf("camera: %s does not support streaming", path)
	}

	// Try MJPEG, then YUYV.
	var actualFmt, actualW, actualH C.uint
	format := C.v4l2_pix_fmt_mjpeg()
	rc := C.v4l2_set_format(fd, format, 640, 480, &actualFmt, &actualW, &actualH)
	if rc < 0 || actualFmt != format {
		format = C.v4l2_pix_fmt_yuyv()
		rc = C.v4l2_set_format(fd, format, 640, 480, &actualFmt, &actualW, &actualH)
		if rc < 0 || actualFmt != format {
			C.v4l2_close(fd)
			return nil, fmt.Errorf("camera: no supported pixel format (tried MJPEG, YUYV)")
		}
	}

	cam := &v4l2Camera{
		fd:     fd,
		format: actualFmt,
		width:  int(actualW),
		height: int(actualH),
	}

	count := C.uint(numBuffers)
	if rc := C.v4l2_request_buffers(fd, &count); rc < 0 {
		C.v4l2_close(fd)
		return nil, fmt.Errorf("camera: REQBUFS: %s", cError(rc))
	}

	cam.buffers = make([]mappedBuffer, count)
	for i := C.uint(0); i < count; i++ {
		var offset, length C.uint
		if rc := C.v4l2_query_buffer(fd, i, &offset, &length); rc < 0 {
			cam.Close()
			return nil, fmt.Errorf("camera: QUERYBUF[%d]: %s", i, cError(rc))
		}
		ptr := C.v4l2_mmap(fd, offset, length)
		if ptr == nil {
			cam.Close()
			return nil, fmt.Errorf("camera: mmap[%d] failed", i)
		}
		cam.buffers[i] = mappedBuffer{ptr: ptr, length: length}
	}

	for i := C.uint(0); i < count; i++ {
		if rc := C.v4l2_queue_buffer(fd, i); rc < 0 {
			cam.Close()
			return nil, fmt.Errorf("camera: QBUF[%d]: %s", i, cError(rc))
		}
	}

	if rc := C.v4l2_stream_on(fd); rc < 0 {
		cam.Close()
		return nil, fmt.Errorf("camera: STREAMON: %s", cError(rc))
	}

	return cam, nil
}

func (c *v4l2Camera) NextFrame() (image.Image, error) {
	r := C.v4l2_poll(c.fd, 5000)
	if r < 0 {
		return nil, fmt.Errorf("camera: poll: %s", cError(r))
	}
	if r == 0 {
		return nil, fmt.Errorf("camera: poll timeout")
	}

	// Drain all available frames, keep the latest.
	var latestData []byte
	type dqd struct{ index, bytesused C.uint }
	var requeue []dqd
	for {
		var index, bytesused C.uint
		if C.v4l2_dequeue_buffer(c.fd, &index, &bytesused) < 0 {
			break
		}
		src := unsafe.Slice((*byte)(c.buffers[index].ptr), bytesused)
		latestData = make([]byte, bytesused)
		copy(latestData, src)
		requeue = append(requeue, dqd{index, bytesused})
	}
	for _, d := range requeue {
		C.v4l2_queue_buffer(c.fd, d.index)
	}

	if latestData == nil {
		return nil, fmt.Errorf("camera: no frame available")
	}

	switch c.format {
	case C.v4l2_pix_fmt_mjpeg():
		return jpeg.Decode(bytes.NewReader(latestData))
	case C.v4l2_pix_fmt_yuyv():
		return yuyvToRGBA(latestData, c.width, c.height), nil
	default:
		return nil, fmt.Errorf("camera: unsupported format 0x%x", c.format)
	}
}

func (c *v4l2Camera) Close() error {
	if c.fd < 0 {
		return nil
	}
	C.v4l2_stream_off(c.fd)
	for _, b := range c.buffers {
		if b.ptr != nil {
			C.v4l2_munmap(b.ptr, b.length)
		}
	}
	c.buffers = nil
	C.v4l2_close(c.fd)
	c.fd = -1
	return nil
}

func yuyvToRGBA(data []byte, width, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := 0; x < width; x += 2 {
			i := (y*width + x) * 2
			if i+3 >= len(data) {
				break
			}
			y0 := float64(data[i])
			u := float64(data[i+1]) - 128
			y1 := float64(data[i+2])
			v := float64(data[i+3]) - 128

			off := (y*width + x) * 4
			img.Pix[off+0] = clampByte(y0 + 1.402*v)
			img.Pix[off+1] = clampByte(y0 - 0.344136*u - 0.714136*v)
			img.Pix[off+2] = clampByte(y0 + 1.772*u)
			img.Pix[off+3] = 255
			img.Pix[off+4] = clampByte(y1 + 1.402*v)
			img.Pix[off+5] = clampByte(y1 - 0.344136*u - 0.714136*v)
			img.Pix[off+6] = clampByte(y1 + 1.772*u)
			img.Pix[off+7] = 255
		}
	}
	return img
}

func clampByte(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func cError(rc C.int) string {
	return fmt.Sprintf("errno %d", -rc)
}
