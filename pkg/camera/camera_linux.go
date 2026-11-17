//go:build linux && cgo

package camera

/*
#include <stdlib.h>

int v4l2_open(const char *path);
int v4l2_close(int fd);
int v4l2_querycap(int fd, char *name, int name_len, unsigned int *caps);
int v4l2_enum_fmt(int fd, unsigned int type, unsigned int index, unsigned int *pixfmt);
int v4l2_set_format(int fd, unsigned int type, unsigned int pixfmt, unsigned int w, unsigned int h,
                    unsigned int *actual_pixfmt, unsigned int *actual_w, unsigned int *actual_h,
                    unsigned int *num_planes, unsigned int *bytesperline);
int v4l2_request_buffers(int fd, unsigned int type, unsigned int *count);
int v4l2_query_buffer(int fd, unsigned int type, unsigned int index, unsigned int num_planes,
                      unsigned int *offsets, unsigned int *lengths);
void *v4l2_mmap(int fd, unsigned int offset, unsigned int length);
int v4l2_munmap(void *addr, unsigned int length);
int v4l2_queue_buffer(int fd, unsigned int type, unsigned int index, unsigned int num_planes);
int v4l2_dequeue_buffer(int fd, unsigned int type, unsigned int num_planes, unsigned int *index,
                        unsigned int *bytesused, unsigned int *data_offset);
int v4l2_stream_on(int fd, unsigned int type);
int v4l2_stream_off(int fd, unsigned int type);
int v4l2_poll(int fd, int timeout_ms);
unsigned int v4l2_cap_video_capture(void);
unsigned int v4l2_cap_video_capture_mplane(void);
unsigned int v4l2_cap_streaming(void);
unsigned int v4l2_buf_type_capture(void);
unsigned int v4l2_buf_type_capture_mplane(void);
unsigned int v4l2_max_planes(void);
*/
import "C"

import (
	"fmt"
	"image"
	"path/filepath"
	"slices"
	"syscall"
	"unsafe"
)

const numBuffers = 4

// maxPlanes mirrors VIDEO_MAX_PLANES; per-plane arrays passed to C use it.
const maxPlanes = 8

func init() {
	if int(C.v4l2_max_planes()) != maxPlanes {
		panic("camera: VIDEO_MAX_PLANES mismatch")
	}
}

type mappedPlane struct {
	ptr    unsafe.Pointer
	length C.uint
}

type v4l2Camera struct {
	fd      C.int
	bufType C.uint
	nplanes C.uint
	format  uint32
	width   int
	height  int
	strides []int
	buffers [][]mappedPlane
}

func list() ([]Device, error) {
	matches, err := filepath.Glob("/dev/video*")
	if err != nil {
		return nil, err
	}
	var devices []Device
	for _, path := range matches {
		var idx int
		if _, err := fmt.Sscanf(filepath.Base(path), "video%d", &idx); err != nil {
			continue
		}
		cpath := C.CString(path)
		fd := C.v4l2_open(cpath)
		C.free(unsafe.Pointer(cpath))
		if fd < 0 {
			continue
		}
		var name [64]C.char
		var caps C.uint
		rc := C.v4l2_querycap(fd, &name[0], 64, &caps)
		if rc < 0 {
			C.v4l2_close(fd)
			continue
		}
		bt, ok := captureBufType(caps)
		usable := ok && len(supportedFormats(fd, bt)) > 0
		C.v4l2_close(fd)
		if !usable {
			continue
		}
		devices = append(devices, Device{
			Index: idx,
			Name:  C.GoString(&name[0]),
			Path:  path,
		})
	}
	slices.SortFunc(devices, func(a, b Device) int { return a.Index - b.Index })
	return devices, nil
}

// captureBufType picks the buffer type for a node's device capabilities.
// Single-planar is preferred when a driver offers both.
func captureBufType(caps C.uint) (C.uint, bool) {
	if caps&C.v4l2_cap_streaming() == 0 {
		return 0, false
	}
	if caps&C.v4l2_cap_video_capture() != 0 {
		return C.v4l2_buf_type_capture(), true
	}
	if caps&C.v4l2_cap_video_capture_mplane() != 0 {
		return C.v4l2_buf_type_capture_mplane(), true
	}
	return 0, false
}

// supportedFormats returns the device formats moxgo can decode, in
// preference order.
func supportedFormats(fd C.int, bt C.uint) []uint32 {
	var have []uint32
	for i := C.uint(0); ; i++ {
		var pf C.uint
		if C.v4l2_enum_fmt(fd, bt, i, &pf) < 0 {
			break
		}
		have = append(have, uint32(pf))
	}
	var out []uint32
	for _, f := range preferredFormats {
		if slices.Contains(have, f) {
			out = append(out, f)
		}
	}
	return out
}

func open(index int, opts Options) (Camera, error) {
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
	bt, ok := captureBufType(caps)
	if !ok {
		C.v4l2_close(fd)
		return nil, fmt.Errorf("camera: %s is not a streaming video capture node (caps 0x%08x)", path, uint32(caps))
	}

	formats := supportedFormats(fd, bt)
	if len(formats) == 0 {
		var have []string
		for i := C.uint(0); ; i++ {
			var pf C.uint
			if C.v4l2_enum_fmt(fd, bt, i, &pf) < 0 {
				break
			}
			have = append(have, fourccString(uint32(pf)))
		}
		C.v4l2_close(fd)
		return nil, fmt.Errorf("camera: %s offers no supported pixel format (device has %v)", path, have)
	}

	cam := &v4l2Camera{fd: fd, bufType: bt}
	var lastErr error
	for _, f := range formats {
		var actualFmt, actualW, actualH, nplanes C.uint
		var bpl [maxPlanes]C.uint
		rc := C.v4l2_set_format(fd, bt, C.uint(f), C.uint(opts.Width), C.uint(opts.Height),
			&actualFmt, &actualW, &actualH, &nplanes, &bpl[0])
		if rc < 0 {
			lastErr = fmt.Errorf("S_FMT %s: %s", fourccString(f), cError(rc))
			continue
		}
		if uint32(actualFmt) != f {
			lastErr = fmt.Errorf("S_FMT %s: driver chose %s", fourccString(f), fourccString(uint32(actualFmt)))
			continue
		}
		cam.format = f
		cam.width, cam.height = int(actualW), int(actualH)
		cam.nplanes = nplanes
		cam.strides = make([]int, nplanes)
		for i := range cam.strides {
			cam.strides[i] = int(bpl[i])
		}
		break
	}
	if cam.format == 0 {
		C.v4l2_close(fd)
		return nil, fmt.Errorf("camera: %s: could not set a pixel format: %v", path, lastErr)
	}

	count := C.uint(numBuffers)
	if rc := C.v4l2_request_buffers(fd, bt, &count); rc < 0 {
		C.v4l2_close(fd)
		return nil, fmt.Errorf("camera: REQBUFS: %s", cError(rc))
	}
	cam.buffers = make([][]mappedPlane, count)
	for i := C.uint(0); i < count; i++ {
		var offsets, lengths [maxPlanes]C.uint
		if rc := C.v4l2_query_buffer(fd, bt, i, cam.nplanes, &offsets[0], &lengths[0]); rc < 0 {
			cam.Close()
			return nil, fmt.Errorf("camera: QUERYBUF[%d]: %s", i, cError(rc))
		}
		cam.buffers[i] = make([]mappedPlane, cam.nplanes)
		for p := range cam.nplanes {
			ptr := C.v4l2_mmap(fd, offsets[p], lengths[p])
			if ptr == nil {
				cam.Close()
				return nil, fmt.Errorf("camera: mmap[%d.%d] failed", i, p)
			}
			cam.buffers[i][p] = mappedPlane{ptr: ptr, length: lengths[p]}
		}
	}

	for i := C.uint(0); i < count; i++ {
		if rc := C.v4l2_queue_buffer(fd, bt, i, cam.nplanes); rc < 0 {
			cam.Close()
			return nil, fmt.Errorf("camera: QBUF[%d]: %s", i, cError(rc))
		}
	}
	if rc := C.v4l2_stream_on(fd, bt); rc < 0 {
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

	// Drain all ready buffers and keep only the newest. Older ones go
	// straight back to the driver.
	latest := -1
	var used, offs [maxPlanes]C.uint
	for {
		var index C.uint
		var u, o [maxPlanes]C.uint
		rc := C.v4l2_dequeue_buffer(c.fd, c.bufType, c.nplanes, &index, &u[0], &o[0])
		if rc < 0 {
			if latest < 0 && rc != -C.int(syscall.EAGAIN) {
				return nil, fmt.Errorf("camera: DQBUF: %s", cError(rc))
			}
			break
		}
		if latest >= 0 {
			C.v4l2_queue_buffer(c.fd, c.bufType, C.uint(latest), c.nplanes)
		}
		latest, used, offs = int(index), u, o
	}
	if latest < 0 {
		return nil, fmt.Errorf("camera: no frame available")
	}

	// Copy out of the mmap'd buffer before requeueing it.
	frame := &rawFrame{
		format:  c.format,
		width:   c.width,
		height:  c.height,
		strides: c.strides,
		planes:  make([][]byte, c.nplanes),
	}
	for p, mp := range c.buffers[latest] {
		start, end := uint(offs[p]), uint(used[p])
		if end == 0 || end > uint(mp.length) {
			end = uint(mp.length)
		}
		if start > end {
			start = end
		}
		src := unsafe.Slice((*byte)(mp.ptr), mp.length)
		frame.planes[p] = append([]byte(nil), src[start:end]...)
	}
	C.v4l2_queue_buffer(c.fd, c.bufType, C.uint(latest), c.nplanes)

	return decodeFrame(frame)
}

func (c *v4l2Camera) Close() error {
	if c.fd < 0 {
		return nil
	}
	C.v4l2_stream_off(c.fd, c.bufType)
	for _, planes := range c.buffers {
		for _, p := range planes {
			if p.ptr != nil {
				C.v4l2_munmap(p.ptr, p.length)
			}
		}
	}
	c.buffers = nil
	C.v4l2_close(c.fd)
	c.fd = -1
	return nil
}

func cError(rc C.int) string {
	return syscall.Errno(-rc).Error()
}

// Describe reports the negotiated format, e.g. "NV12 1280x720 (mplane, 2 planes)".
func (c *v4l2Camera) Describe() string {
	api := "single-planar"
	if c.bufType == C.v4l2_buf_type_capture_mplane() {
		api = fmt.Sprintf("mplane, %d planes", c.nplanes)
	}
	return fmt.Sprintf("%s %dx%d (%s)", fourccString(c.format), c.width, c.height, api)
}
