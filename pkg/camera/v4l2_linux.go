//go:build linux && (amd64 || arm64)

package camera

import (
	"errors"
	"fmt"
	"image"
	"path/filepath"
	"runtime"
	"slices"
	"syscall"
	"time"
	"unsafe"

	"github.com/aep/gaal"
)

// V4L2 capture, spoken directly to the kernel: no C. The numbers below are
// from linux/videodev2.h for 64-bit kernels and user space.
const (
	vidiocQueryCap = 0x80685600
	vidiocEnumFmt  = 0xc0405602
	vidiocSFmt     = 0xc0d05605
	vidiocReqBufs  = 0xc0145608
	vidiocQueryBuf = 0xc0585609
	vidiocQBuf     = 0xc058560f
	vidiocDQBuf    = 0xc0585611
	vidiocStreamOn = 0x40045612
	vidiocStreamOf = 0x40045613
	vidiocSParm    = 0xc0cc5616

	capCapture       = 0x00000001
	capCaptureMPlane = 0x00001000
	capStreaming     = 0x04000000
	capDeviceCaps    = 0x80000000

	bufTypeCapture       = 1
	bufTypeCaptureMPlane = 9
	memoryMmap           = 1
	fieldNone            = 1

	bufFlagTimestampMask = 0xe000
	bufFlagMonotonic     = 0x2000

	capStructSize   = 104
	fmtDescSize     = 64
	formatSize      = 208
	parmSize        = 204
	reqBufsSize     = 20
	bufferSize      = 88
	planeSize       = 64
	planeFmtSize    = 20
	fmtPixFormat    = 16
	fmtPixStride    = 24
	fmtMPlaneFmts   = 28
	fmtMPlaneCount  = 188
	planeDataOffset = 16
	bufOffsetSec    = 24
	bufOffsetUsec   = 32
	bufOffsetSeq    = 56
	bufOffsetMemory = 60
	bufOffsetM      = 64
	bufOffsetLength = 72
)

const numBuffers = 4

// maxPlanes mirrors VIDEO_MAX_PLANES.
const maxPlanes = 8

func get32(b []byte, off int) uint32 { return *(*uint32)(unsafe.Pointer(&b[off])) }
func put32(b []byte, off int, v uint32) {
	*(*uint32)(unsafe.Pointer(&b[off])) = v
}
func get64(b []byte, off int) uint64 { return *(*uint64)(unsafe.Pointer(&b[off])) }
func put64(b []byte, off int, v uint64) {
	*(*uint64)(unsafe.Pointer(&b[off])) = v
}

func ioctl(fd int, req uintptr, arg []byte) error {
	var p unsafe.Pointer
	if len(arg) > 0 {
		p = unsafe.Pointer(&arg[0])
	}
	for {
		_, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, uintptr(p))
		if e == syscall.EINTR {
			continue
		}
		runtime.KeepAlive(arg)
		if e != 0 {
			return e
		}
		return nil
	}
}

func openNode(path string) (int, error) {
	return syscall.Open(path, syscall.O_RDWR|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
}

// queryCap returns the card name and the capabilities of the opened node.
func queryCap(fd int) (string, uint32, error) {
	var c [capStructSize]byte
	if err := ioctl(fd, vidiocQueryCap, c[:]); err != nil {
		return "", 0, err
	}
	name := c[16:48]
	if i := slices.Index(name, 0); i >= 0 {
		name = name[:i]
	}
	caps := get32(c[:], 84)
	if caps&capDeviceCaps != 0 {
		caps = get32(c[:], 88)
	}
	return string(name), caps, nil
}

// captureBufType picks the buffer type for a node's device capabilities.
// Single-planar is preferred when a driver offers both.
func captureBufType(caps uint32) (uint32, bool) {
	if caps&capStreaming == 0 {
		return 0, false
	}
	if caps&capCapture != 0 {
		return bufTypeCapture, true
	}
	if caps&capCaptureMPlane != 0 {
		return bufTypeCaptureMPlane, true
	}
	return 0, false
}

func deviceFormats(fd int, bt uint32) []uint32 {
	var have []uint32
	for i := uint32(0); ; i++ {
		var d [fmtDescSize]byte
		put32(d[:], 0, i)
		put32(d[:], 4, bt)
		if err := ioctl(fd, vidiocEnumFmt, d[:]); err != nil {
			return have
		}
		have = append(have, get32(d[:], 44))
	}
}

// supportedFormats returns the device formats moxgo can decode, in
// preference order.
func supportedFormats(fd int, bt uint32) []uint32 {
	have := deviceFormats(fd, bt)
	var out []uint32
	for _, f := range preferredFormats {
		if slices.Contains(have, f) {
			out = append(out, f)
		}
	}
	return out
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
		fd, err := openNode(path)
		if err != nil {
			continue
		}
		name, caps, err := queryCap(fd)
		usable := false
		if err == nil {
			if bt, ok := captureBufType(caps); ok {
				usable = len(supportedFormats(fd, bt)) > 0
			}
		}
		syscall.Close(fd)
		if usable {
			devices = append(devices, Device{Index: idx, Name: name, Path: path})
		}
	}
	slices.SortFunc(devices, func(a, b Device) int { return a.Index - b.Index })
	return devices, nil
}

// v4l2Device is an open, streaming capture node.
type v4l2Device struct {
	fd      int
	path    string
	card    string
	bufType uint32
	nplanes int
	format  uint32
	width   int
	height  int
	strides []int
	maps    [][][]byte // buffer, plane

	buf    [bufferSize]byte
	planes []byte // v4l2_plane array, on the heap so the kernel can write it
	poll   [8]byte

	monoBase time.Time // wall/monotonic time at monoZero
	monoZero time.Duration
}

func monotonicNow() time.Duration {
	var ts syscall.Timespec
	// CLOCK_MONOTONIC is 1.
	syscall.Syscall(syscall.SYS_CLOCK_GETTIME, 1, uintptr(unsafe.Pointer(&ts)), 0)
	return time.Duration(ts.Sec)*time.Second + time.Duration(ts.Nsec)
}

func openDevice(index int, opts Options) (*v4l2Device, error) {
	path := fmt.Sprintf("/dev/video%d", index)
	fd, err := openNode(path)
	if err != nil {
		return nil, fmt.Errorf("camera: open %s: %w", path, err)
	}
	d := &v4l2Device{fd: fd, path: path, monoBase: time.Now(), monoZero: monotonicNow()}
	if err := d.setup(opts); err != nil {
		d.close()
		return nil, err
	}
	return d, nil
}

func (d *v4l2Device) setup(opts Options) error {
	card, caps, err := queryCap(d.fd)
	if err != nil {
		return fmt.Errorf("camera: QUERYCAP: %w", err)
	}
	d.card = card
	bt, ok := captureBufType(caps)
	if !ok {
		return fmt.Errorf("camera: %s is not a streaming video capture node (caps 0x%08x)", d.path, caps)
	}
	d.bufType = bt
	mplane := bt == bufTypeCaptureMPlane

	formats := supportedFormats(d.fd, bt)
	if len(formats) == 0 {
		var have []string
		for _, f := range deviceFormats(d.fd, bt) {
			have = append(have, fourccString(f))
		}
		return fmt.Errorf("camera: %s offers no supported pixel format (device has %v)", d.path, have)
	}
	if opts.Format != 0 {
		if !slices.Contains(formats, opts.Format) {
			return fmt.Errorf("camera: %s does not offer %s", d.path, fourccString(opts.Format))
		}
		formats = []uint32{opts.Format}
	}

	var lastErr error
	for _, f := range formats {
		var fm [formatSize]byte
		put32(fm[:], 0, bt)
		put32(fm[:], 8, uint32(opts.Width))
		put32(fm[:], 12, uint32(opts.Height))
		put32(fm[:], fmtPixFormat, f)
		put32(fm[:], 20, fieldNone)
		if err := ioctl(d.fd, vidiocSFmt, fm[:]); err != nil {
			lastErr = fmt.Errorf("S_FMT %s: %w", fourccString(f), err)
			continue
		}
		if got := get32(fm[:], fmtPixFormat); got != f {
			lastErr = fmt.Errorf("S_FMT %s: driver chose %s", fourccString(f), fourccString(got))
			continue
		}
		d.format = f
		d.width, d.height = int(get32(fm[:], 8)), int(get32(fm[:], 12))
		if mplane {
			d.nplanes = clampInt(int(fm[fmtMPlaneCount]), 1, maxPlanes)
			d.strides = make([]int, d.nplanes)
			for i := range d.strides {
				d.strides[i] = int(get32(fm[:], fmtMPlaneFmts+i*planeFmtSize+4))
			}
		} else {
			d.nplanes = 1
			d.strides = []int{int(get32(fm[:], fmtPixStride))}
		}
		break
	}
	if d.format == 0 {
		return fmt.Errorf("camera: %s: could not set a pixel format: %v", d.path, lastErr)
	}

	if opts.FPS > 0 {
		var parm [parmSize]byte
		put32(parm[:], 0, bt)
		put32(parm[:], 12, 1)
		put32(parm[:], 16, uint32(opts.FPS))
		_ = ioctl(d.fd, vidiocSParm, parm[:]) // best effort
	}

	var req [reqBufsSize]byte
	put32(req[:], 0, numBuffers)
	put32(req[:], 4, bt)
	put32(req[:], 8, memoryMmap)
	if err := ioctl(d.fd, vidiocReqBufs, req[:]); err != nil {
		return fmt.Errorf("camera: REQBUFS: %w", err)
	}
	count := int(get32(req[:], 0))
	d.planes = make([]byte, maxPlanes*planeSize)
	d.maps = make([][][]byte, count)
	for i := 0; i < count; i++ {
		d.prepareBuf(i)
		if err := ioctl(d.fd, vidiocQueryBuf, d.buf[:]); err != nil {
			return fmt.Errorf("camera: QUERYBUF[%d]: %w", i, err)
		}
		d.maps[i] = make([][]byte, d.nplanes)
		for p := 0; p < d.nplanes; p++ {
			var off int64
			var length int
			if mplane {
				pl := d.planes[p*planeSize:]
				length, off = int(get32(pl, 4)), int64(get32(pl, 8))
			} else {
				off, length = int64(get32(d.buf[:], bufOffsetM)), int(get32(d.buf[:], bufOffsetLength))
			}
			m, err := syscall.Mmap(d.fd, off, length, syscall.PROT_READ, syscall.MAP_SHARED)
			if err != nil {
				return fmt.Errorf("camera: mmap[%d.%d]: %w", i, p, err)
			}
			d.maps[i][p] = m
		}
	}
	for i := 0; i < count; i++ {
		if err := d.queue(i); err != nil {
			return fmt.Errorf("camera: QBUF[%d]: %w", i, err)
		}
	}
	kind := make([]byte, 4)
	put32(kind, 0, bt)
	if err := ioctl(d.fd, vidiocStreamOn, kind); err != nil {
		return fmt.Errorf("camera: STREAMON: %w", err)
	}
	return nil
}

// prepareBuf fills d.buf for buffer i, or for a dequeue when i is negative.
func (d *v4l2Device) prepareBuf(i int) {
	b := d.buf[:]
	clear(b)
	if i >= 0 {
		put32(b, 0, uint32(i))
	}
	put32(b, 4, d.bufType)
	put32(b, bufOffsetMemory, memoryMmap)
	if d.bufType == bufTypeCaptureMPlane {
		clear(d.planes)
		put64(b, bufOffsetM, uint64(uintptr(unsafe.Pointer(&d.planes[0]))))
		put32(b, bufOffsetLength, uint32(d.nplanes))
	}
}

func (d *v4l2Device) queue(i int) error {
	d.prepareBuf(i)
	err := ioctl(d.fd, vidiocQBuf, d.buf[:])
	runtime.KeepAlive(d.planes)
	return err
}

func (d *v4l2Device) wait(timeout time.Duration) error {
	// struct pollfd: int fd, short events, short revents.
	const pollIn = 1
	put32(d.poll[:], 0, uint32(d.fd))
	put32(d.poll[:], 4, pollIn)
	ts := syscall.NsecToTimespec(int64(timeout))
	n, _, e := syscall.Syscall6(syscall.SYS_PPOLL, uintptr(unsafe.Pointer(&d.poll)), 1, uintptr(unsafe.Pointer(&ts)), 0, 0, 0)
	if e != 0 && e != syscall.EINTR {
		return fmt.Errorf("camera: poll: %w", e)
	}
	if n == 0 && e == 0 {
		return ErrTimeout
	}
	return nil
}

// capture is a dequeued buffer; release gives it back to the driver.
type capture struct {
	index    int
	data     [maxPlanes][]byte
	captured time.Time // when the driver stamped it; zero if it does not
	received time.Time // when it was dequeued
	seq      uint32
}

// dequeue waits for a frame and returns the newest one ready. Older ones go
// straight back to the driver.
func (d *v4l2Device) dequeue(timeout time.Duration, out *capture) error {
	if err := d.wait(timeout); err != nil {
		return err
	}
	got := false
	for {
		d.prepareBuf(-1)
		err := ioctl(d.fd, vidiocDQBuf, d.buf[:])
		runtime.KeepAlive(d.planes)
		if err != nil {
			if errors.Is(err, syscall.EAGAIN) {
				if got {
					return nil
				}
				return errAgain
			}
			if got {
				return nil
			}
			return fmt.Errorf("camera: DQBUF: %w", err)
		}
		// Read this frame before queue overwrites d.buf.
		var next capture
		d.fill(&next)
		if got {
			if err := d.queue(out.index); err != nil {
				return fmt.Errorf("camera: QBUF: %w", err)
			}
		}
		*out = next
		got = true
	}
}

func (d *v4l2Device) fill(out *capture) {
	b := d.buf[:]
	i := int(get32(b, 0))
	out.index = i
	out.seq = get32(b, bufOffsetSeq)
	out.received = time.Now()
	out.captured = time.Time{}
	if get32(b, 12)&bufFlagTimestampMask == bufFlagMonotonic {
		ts := time.Duration(get64(b, bufOffsetSec))*time.Second + time.Duration(get64(b, bufOffsetUsec))*time.Microsecond
		if ts > 0 {
			out.captured = d.monoBase.Add(ts - d.monoZero)
		}
	}
	for p := 0; p < d.nplanes; p++ {
		m := d.maps[i][p]
		start, end := 0, len(m)
		if d.bufType == bufTypeCaptureMPlane {
			pl := d.planes[p*planeSize:]
			start, end = int(get32(pl, planeDataOffset)), int(get32(pl, 0))
		} else if used := int(get32(b, 8)); used > 0 {
			end = used
		}
		if end <= 0 || end > len(m) {
			end = len(m)
		}
		if start > end {
			start = end
		}
		out.data[p] = m[start:end]
	}
}

func (d *v4l2Device) raw(c *capture) rawFrame {
	return rawFrame{format: d.format, width: d.width, height: d.height, strides: d.strides, planes: c.data[:d.nplanes]}
}

func (d *v4l2Device) close() {
	if d.fd < 0 {
		return
	}
	kind := make([]byte, 4)
	put32(kind, 0, d.bufType)
	_ = ioctl(d.fd, vidiocStreamOf, kind)
	for _, planes := range d.maps {
		for _, m := range planes {
			if m != nil {
				syscall.Munmap(m)
			}
		}
	}
	d.maps = nil
	syscall.Close(d.fd)
	d.fd = -1
}

// describe reports the negotiated format, e.g. "NV12 1280x720 (mplane, 2 planes)".
func (d *v4l2Device) describe() string {
	api := "single-planar"
	if d.bufType == bufTypeCaptureMPlane {
		api = fmt.Sprintf("mplane, %d planes", d.nplanes)
	}
	return fmt.Sprintf("%s %dx%d (%s)", fourccString(d.format), d.width, d.height, api)
}

// v4l2Camera is the image.Image interface over a device.
type v4l2Camera struct {
	d   *v4l2Device
	cap capture
}

func open(index int, opts Options) (Camera, error) {
	d, err := openDevice(index, opts)
	if err != nil {
		return nil, err
	}
	return &v4l2Camera{d: d}, nil
}

func (c *v4l2Camera) NextFrame() (image.Image, error) {
	deadline := time.Now().Add(5 * time.Second)
	for {
		left := time.Until(deadline)
		if left <= 0 {
			return nil, ErrTimeout
		}
		err := c.d.dequeue(left, &c.cap)
		if errors.Is(err, errAgain) {
			continue
		}
		if err != nil {
			return nil, err
		}
		break
	}
	f := c.d.raw(&c.cap)
	// decodeFrame copies out of the mmap'd buffer before it is requeued.
	img, derr := decodeFrame(&f)
	if err := c.d.queue(c.cap.index); err != nil && derr == nil {
		derr = fmt.Errorf("camera: QBUF: %w", err)
	}
	return img, derr
}

func (c *v4l2Camera) Close() error {
	c.d.close()
	return nil
}

// Describe reports the negotiated format.
func (c *v4l2Camera) Describe() string { return c.d.describe() }

// v4l2Source delivers a device's frames as gaal frames, decoding MJPEG with
// the best backend available.
type v4l2Source struct {
	d       *v4l2Device
	lib     *gaal.Library
	decoder gaal.VideoDecoder
	frames  gaal.FramePool
	cap     capture
	name    string
}

func openV4L2Source(index int, cfg SourceConfig) (VideoSource, error) {
	d, err := openDevice(index, Options{Width: cfg.Width, Height: cfg.Height, FPS: cfg.FPS, Format: cfg.pixelFormat()})
	if err != nil {
		return nil, err
	}
	s := &v4l2Source{d: d, lib: cfg.library()}
	s.name = fmt.Sprintf("%s [%s] %s", d.path, d.card, d.describe())
	if d.format == pixFmtMJPEG || d.format == pixFmtJPEG {
		dec, err := s.lib.NewVideoDecoder(gaal.VideoConfig{Codec: gaal.MJPEG})
		if err != nil {
			d.close()
			return nil, fmt.Errorf("camera: MJPEG decoder: %w", err)
		}
		s.decoder = dec
		s.name += " via " + dec.Backend().Name
	}
	return s, nil
}

func (s *v4l2Source) Name() string { return s.name }

// Matrix implements VideoSource: JPEG and webcam YUV are BT.601.
func (s *v4l2Source) Matrix() ColorMatrix { return MatrixBT601 }

func (s *v4l2Source) Next(timeout time.Duration) (*gaal.VideoFrame, FrameTiming, error) {
	deadline := time.Now().Add(timeout)
	for {
		left := time.Until(deadline)
		if left <= 0 {
			return nil, FrameTiming{}, ErrTimeout
		}
		err := s.d.dequeue(left, &s.cap)
		if errors.Is(err, errAgain) {
			continue
		}
		if err != nil {
			return nil, FrameTiming{}, err
		}
		t := FrameTiming{Captured: s.cap.captured, Received: s.cap.received}
		if t.Captured.IsZero() {
			t.Captured = t.Received
		}
		raw := s.d.raw(&s.cap)
		var f *gaal.VideoFrame
		if s.decoder != nil {
			t.Sent = time.Now()
			if len(raw.planes[0]) == 0 {
				_ = s.d.queue(s.cap.index)
				continue
			}
			err = s.decoder.Send(gaal.Packet{Data: raw.planes[0], Keyframe: true})
			if qerr := s.d.queue(s.cap.index); qerr != nil {
				return nil, t, fmt.Errorf("camera: QBUF: %w", qerr)
			}
			if err == nil {
				f, err = gaal.ReceiveWait(s.decoder, max(left, time.Second))
			}
			if errors.Is(err, gaal.ErrCorrupt) || errors.Is(err, gaal.ErrInvalid) || errors.Is(err, gaal.ErrAgain) {
				continue // cameras emit damaged JPEGs now and then
			}
			if err != nil {
				return nil, t, err
			}
			if f.Range == gaal.RangeUnspecified {
				f.Range = gaal.RangeFull // JFIF
			}
		} else {
			t.Sent = time.Now()
			f, err = rawToVideoFrame(&raw, &s.frames)
			if qerr := s.d.queue(s.cap.index); qerr != nil {
				if f != nil {
					f.Release()
				}
				return nil, t, fmt.Errorf("camera: QBUF: %w", qerr)
			}
			if err != nil {
				return nil, t, err
			}
		}
		f.PTS = t.Captured.Sub(s.d.monoBase)
		t.Decoded = time.Now()
		return f, t, nil
	}
}

func (s *v4l2Source) Close() error {
	s.d.close()
	if s.decoder != nil {
		s.decoder.Close()
		s.decoder = nil
	}
	return nil
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
