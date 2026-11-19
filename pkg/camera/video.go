package camera

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/aep/gaal"
	gcam "github.com/aep/gaal/camera"
	"github.com/aep/gaal/onvif"
	"github.com/aep/gaal/rtsp"
)

// Environment variables with the credentials of cameras whose URL has none.
const (
	EnvUser     = "MOXGO_CAMERA_USER"
	EnvPassword = "MOXGO_CAMERA_PASSWORD"
)

// ColorMatrix is the YCbCr matrix of a source's frames.
type ColorMatrix int

// Color matrices. MatrixAuto is BT.709 for HD sizes and BT.601 below.
const (
	MatrixAuto ColorMatrix = iota
	MatrixBT601
	MatrixBT709
)

// FrameTiming says when a frame passed the stages before the pipeline. Any
// field but Decoded can be zero when a source cannot know it.
type FrameTiming struct {
	// Captured is the earliest moment known for the picture: the driver's
	// capture timestamp of a local camera, the network arrival of a stream.
	Captured time.Time
	Received time.Time // the data was available to this process
	Sent     time.Time // the data went to the decoder
	Decoded  time.Time // the decoder returned the frame
}

// VideoSource delivers decoded frames as fast as they arrive.
type VideoSource interface {
	// Next returns the next frame, which the caller must Release. It returns
	// an error wrapping ErrTimeout if none arrives in time.
	Next(timeout time.Duration) (*gaal.VideoFrame, FrameTiming, error)
	// Name describes the source, without credentials.
	Name() string
	Matrix() ColorMatrix
	Close() error
}

// SourceConfig configures OpenVideoCamera, OpenVideoStream and OpenONVIF.
// Zero values mean defaults.
type SourceConfig struct {
	Width, Height int // camera capture size, default 640x480
	FPS           int // camera frame rate, default the driver's
	Format        string
	Backend       string        // gaal decoder backend
	Library       *gaal.Library // default: gaal.New
	// Transport is the RTSP transport: "tcp" (default), "udp" or "auto".
	Transport string
	// KeepDecoderDelay leaves H.264 parameter sets as they are.
	KeepDecoderDelay bool
	// Username and Password override those in the URL and the environment.
	Username, Password string
}

func (c SourceConfig) library() *gaal.Library {
	if c.Library != nil {
		return c.Library
	}
	return gaal.New(gaal.Options{Backend: c.Backend})
}

func (c SourceConfig) pixelFormat() uint32 {
	if len(c.Format) != 4 {
		return 0
	}
	return fourcc(c.Format[0], c.Format[1], c.Format[2], c.Format[3])
}

// --- credentials and redaction ---

var (
	secretsMu sync.RWMutex
	secrets   []string
)

// registerSecret makes Redact remove v and its encodings.
func registerSecret(user, pass string) {
	var add []string
	if pass != "" {
		add = append(add, pass, url.QueryEscape(pass), url.PathEscape(pass),
			base64.StdEncoding.EncodeToString([]byte(user+":"+pass)))
	}
	if len(user) >= 3 {
		add = append(add, user, url.QueryEscape(user))
	}
	secretsMu.Lock()
	defer secretsMu.Unlock()
	for _, s := range add {
		dup := false
		for _, x := range secrets {
			dup = dup || x == s
		}
		if !dup {
			secrets = append(secrets, s)
		}
	}
	// Longest first, so an encoded form is not left half replaced.
	for i := range secrets {
		for j := i + 1; j < len(secrets); j++ {
			if len(secrets[j]) > len(secrets[i]) {
				secrets[i], secrets[j] = secrets[j], secrets[i]
			}
		}
	}
}

// Redact removes credentials from text meant for output: the user
// information and credential-like query parameters of every URL in it, and
// all credentials this process has used, in plain or encoded form.
func Redact(s string) string {
	s = rtsp.RedactText(s)
	secretsMu.RLock()
	defer secretsMu.RUnlock()
	for _, x := range secrets {
		s = strings.ReplaceAll(s, x, "***")
	}
	return s
}

// RedactError is Redact for an error; nil stays nil.
func RedactError(err error) error {
	if err == nil {
		return nil
	}
	return &redactedError{msg: Redact(err.Error()), err: err}
}

type redactedError struct {
	msg string
	err error
}

func (e *redactedError) Error() string        { return e.msg }
func (e *redactedError) Is(target error) bool { return errors.Is(e.err, target) }

// Credentials splits raw into a URL without user information and the user
// and password to use: explicit ones first, then the URL's, then the
// environment. They are registered with Redact.
func Credentials(raw, user, pass string) (clean, u, p string) {
	clean = raw
	if parsed, err := url.Parse(raw); err == nil && parsed.User != nil {
		if user == "" {
			user = parsed.User.Username()
		}
		if pw, ok := parsed.User.Password(); ok && pass == "" {
			pass = pw
		}
		parsed.User = nil
		clean = parsed.String()
	}
	if user == "" {
		user = os.Getenv(EnvUser)
	}
	if pass == "" {
		pass = os.Getenv(EnvPassword)
	}
	registerSecret(user, pass)
	return clean, user, pass
}

// --- opening ---

// OpenVideoCamera opens local camera index as a frame source.
func OpenVideoCamera(index int, cfg SourceConfig) (VideoSource, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		cfg.Width, cfg.Height = 640, 480
	}
	src, err := openPlatformSource(index, cfg)
	if err != nil {
		return nil, err
	}
	return src, nil
}

func isRTSP(u string) bool { return strings.HasPrefix(strings.ToLower(u), "rtsp://") }

// isNoDecoder reports that gaal has no way to decode what was asked.
func isNoDecoder(err error) bool {
	var nb *gaal.NoBackendError
	return errors.As(err, &nb)
}

// OpenVideoStream opens a stream URL: rtsp:// through gaal, http(s):// as
// MJPEG, anything else (and RTSP gaal cannot decode) through ffmpeg.
func OpenVideoStream(raw string, cfg SourceConfig) (VideoSource, error) {
	clean, user, pass := Credentials(raw, cfg.Username, cfg.Password)
	switch {
	case isRTSP(clean):
		src, err := openRTSP(clean, user, pass, cfg)
		if err == nil {
			return src, nil
		}
		if !isNoDecoder(err) {
			return nil, RedactError(err)
		}
		fallthrough
	default:
		// ffmpeg takes credentials in the URL only.
		ffURL := raw
		if parsed, perr := url.Parse(raw); perr == nil && parsed.User == nil && user != "" {
			parsed.User = url.UserPassword(user, pass)
			ffURL = parsed.String()
		}
		cam, err := openFFmpegStream(ffURL)
		if err != nil {
			return nil, RedactError(err)
		}
		return newImageSource(cam, "ffmpeg "+Redact(clean)), nil
	case strings.HasPrefix(clean, "http://"), strings.HasPrefix(clean, "https://"):
		src, err := openHTTPSource(clean, user, pass, cfg)
		return src, RedactError(err)
	}
}

// --- RTSP ---

// transport maps a --rtsp-transport value.
func rtspTransport(s string) (rtsp.Transport, error) {
	switch strings.ToLower(s) {
	case "", "tcp":
		return rtsp.TransportTCP, nil
	case "udp":
		return rtsp.TransportUDP, nil
	case "auto":
		return rtsp.TransportAuto, nil
	}
	return 0, fmt.Errorf("camera: unknown RTSP transport %q (want tcp, udp or auto)", s)
}

// maxOrderViolations is how many frames may come out of the decoder with
// decreasing timestamps before the low-delay rewrite is taken back.
const maxOrderViolations = 3

type rtspSource struct {
	cfg        rtsp.Config
	lib        *gaal.Library
	keepDelay  bool
	cam        *gcam.RTSPCamera
	lastPTS    time.Duration
	havePTS    bool
	violations int
	// OutOfOrder counts frames decoded with a timestamp not after the one
	// before, over the whole session.
	outOfOrder uint64
	rng        rangeProbe
}

// rangeProbe decides the range of a stream that does not state it. Many IP
// cameras send full-range video without saying so, and reading it as limited
// range clips shadows and highlights. Samples well outside 16..235 settle it.
type rangeProbe struct {
	full   bool
	frames int
}

// outOfRangeShare is the share of luma samples outside the limited range that
// marks a stream as full range.
const outOfRangeShare = 0.005

// apply sets f.Range when it is unspecified.
func (p *rangeProbe) apply(f *gaal.VideoFrame) {
	if f.Range != gaal.RangeUnspecified {
		return
	}
	if !p.full && p.frames < 300 {
		p.frames++
		out, n := 0, 0
		for y := 0; y < f.Height; y += 8 {
			row := f.Y[y*f.YStride : y*f.YStride+f.Width]
			for x := 0; x < len(row); x += 8 {
				n++
				if v := row[x]; v < 16 || v > 235 {
					out++
				}
			}
		}
		if n > 0 && float64(out) > outOfRangeShare*float64(n) {
			p.full = true
		}
	}
	if p.full {
		f.Range = gaal.RangeFull
	} else {
		f.Range = gaal.RangeLimited
	}
}

func openRTSP(clean, user, pass string, cfg SourceConfig) (VideoSource, error) {
	tr, err := rtspTransport(cfg.Transport)
	if err != nil {
		return nil, err
	}
	s := &rtspSource{
		lib:       cfg.library(),
		keepDelay: cfg.KeepDecoderDelay,
		cfg: rtsp.Config{
			URL: clean, Username: user, Password: pass,
			Transport: tr,
			Reconnect: &rtsp.Reconnect{MinDelay: time.Second, MaxDelay: 10 * time.Second},
		},
	}
	if err := s.open(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *rtspSource) open() error {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cam, err := gcam.OpenRTSP(ctx, gcam.RTSPConfig{RTSP: s.cfg, KeepDecoderDelay: s.keepDelay, Library: s.lib})
	if err != nil {
		return err
	}
	s.cam = cam
	s.havePTS = false
	return nil
}

func (s *rtspSource) Name() string        { return Redact(s.cam.Name()) }
func (s *rtspSource) Matrix() ColorMatrix { return MatrixAuto }

// OutOfOrder is the number of frames decoded out of timestamp order.
func (s *rtspSource) OutOfOrder() uint64 { return s.outOfOrder }

func (s *rtspSource) Next(timeout time.Duration) (*gaal.VideoFrame, FrameTiming, error) {
	f, t, err := s.cam.NextTimed(timeout)
	if err != nil {
		if errors.Is(err, rtsp.ErrTimeout) {
			return nil, FrameTiming{}, fmt.Errorf("%w: %s", ErrTimeout, Redact(err.Error()))
		}
		return nil, FrameTiming{}, RedactError(err)
	}
	if s.havePTS && f.PTS <= s.lastPTS {
		s.outOfOrder++
		s.violations++
		// The stream does reorder after all: LowDelay was wrong for it. Keep
		// the order right, paying the decoder's delay.
		if !s.keepDelay && s.violations >= maxOrderViolations {
			s.keepDelay = true
			s.cam.Close()
			if err := s.open(); err != nil {
				f.Release()
				return nil, FrameTiming{}, RedactError(err)
			}
		}
	}
	s.lastPTS, s.havePTS = f.PTS, true
	s.rng.apply(f)
	ft := FrameTiming{Captured: t.Arrival, Received: t.Arrival, Sent: t.Sent, Decoded: t.Decoded}
	if ft.Captured.IsZero() {
		ft.Captured = t.Decoded
		ft.Received = t.Decoded
	}
	return f, ft, nil
}

func (s *rtspSource) Close() error { return s.cam.Close() }

// --- ONVIF ---

// Profile is a media profile of an ONVIF device.
type Profile struct {
	Token, Name string
	Codec       string
	Width       int
	Height      int
	FPS         float64
}

func onvifClient(service, user, pass string) (*onvif.Client, error) {
	return onvif.New(onvif.Config{URL: service, Username: user, Password: pass})
}

// normalizeONVIF accepts "host", "host:port" or a device service URL.
func normalizeONVIF(s string) string {
	if strings.Contains(s, "://") {
		return s
	}
	return "http://" + s + "/onvif/device_service"
}

// ONVIFProfiles lists the media profiles of a device.
func ONVIFProfiles(ctx context.Context, service string, cfg SourceConfig) ([]Profile, error) {
	clean, user, pass := Credentials(normalizeONVIF(service), cfg.Username, cfg.Password)
	c, err := onvifClient(clean, user, pass)
	if err != nil {
		return nil, RedactError(err)
	}
	ps, err := c.Profiles(ctx)
	if err != nil {
		return nil, RedactError(err)
	}
	out := make([]Profile, 0, len(ps))
	for _, p := range ps {
		e := Profile{Token: p.Token, Name: p.Name}
		if v := p.Video; v != nil {
			e.Codec, e.Width, e.Height, e.FPS = v.Encoding, v.Width, v.Height, v.FrameRate
		}
		out = append(out, e)
	}
	return out, nil
}

// OpenONVIF opens a stream of an ONVIF device. profile is a profile name or
// token, "main" for the largest, "sub" for the smallest; empty means main.
func OpenONVIF(ctx context.Context, service, profile string, cfg SourceConfig) (VideoSource, error) {
	clean, user, pass := Credentials(normalizeONVIF(service), cfg.Username, cfg.Password)
	c, err := onvifClient(clean, user, pass)
	if err != nil {
		return nil, RedactError(err)
	}
	ps, err := c.Profiles(ctx)
	if err != nil {
		return nil, RedactError(err)
	}
	p, err := pickProfile(ps, profile)
	if err != nil {
		return nil, RedactError(err)
	}
	uri, err := c.StreamURI(ctx, p)
	if err != nil {
		return nil, RedactError(err)
	}
	cfg.Username, cfg.Password = user, pass
	return OpenVideoStream(uri, cfg)
}

func pickProfile(ps []onvif.Profile, want string) (onvif.Profile, error) {
	if len(ps) == 0 {
		return onvif.Profile{}, errors.New("camera: the device has no media profiles")
	}
	area := func(p onvif.Profile) int {
		if p.Video == nil {
			return 0
		}
		return p.Video.Width * p.Video.Height
	}
	best := ps[0]
	switch strings.ToLower(want) {
	case "", "main":
		for _, p := range ps {
			if area(p) > area(best) {
				best = p
			}
		}
		return best, nil
	case "sub":
		for _, p := range ps {
			if area(p) > 0 && (area(best) == 0 || area(p) < area(best)) {
				best = p
			}
		}
		return best, nil
	}
	for _, p := range ps {
		if p.Token == want || strings.EqualFold(p.Name, want) {
			return p, nil
		}
	}
	return onvif.Profile{}, fmt.Errorf("camera: no ONVIF profile %q", want)
}

// DiscoveredDevice is an ONVIF device found on the network.
type DiscoveredDevice struct {
	Name, Hardware, Address string
	Services                []string
}

// DiscoverONVIF probes the local network for ONVIF devices.
func DiscoverONVIF(ctx context.Context, timeout time.Duration) ([]DiscoveredDevice, error) {
	found, err := onvif.Discover(ctx, onvif.DiscoverOptions{Timeout: timeout})
	if err != nil {
		return nil, RedactError(err)
	}
	out := make([]DiscoveredDevice, 0, len(found))
	for _, d := range found {
		svc := make([]string, len(d.XAddrs))
		for i, x := range d.XAddrs {
			svc[i] = Redact(x)
		}
		out = append(out, DiscoveredDevice{Name: d.Name(), Hardware: d.Hardware(), Address: d.From, Services: svc})
	}
	return out, nil
}

// --- HTTP MJPEG ---

type httpSource struct {
	name    string
	resp    *http.Response
	decoder gaal.VideoDecoder

	mu      sync.Mutex
	pending *jpegPart
	spare   *jpegPart
	ready   chan struct{}
	done    chan struct{}
	err     error
}

type jpegPart struct {
	data    bytes.Buffer
	arrived time.Time
}

func openHTTPSource(clean, user, pass string, cfg SourceConfig) (VideoSource, error) {
	req, err := http.NewRequest(http.MethodGet, clean, nil)
	if err != nil {
		return nil, fmt.Errorf("stream: %w", err)
	}
	if user != "" {
		req.SetBasicAuth(user, pass)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("stream: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("stream: HTTP %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	mediaType, params, _ := mime.ParseMediaType(ct)
	if mediaType != "multipart/x-mixed-replace" || params["boundary"] == "" {
		resp.Body.Close()
		return nil, fmt.Errorf("stream: unsupported Content-Type %q (expected multipart/x-mixed-replace with a boundary)", ct)
	}
	dec, err := cfg.library().NewVideoDecoder(gaal.VideoConfig{Codec: gaal.MJPEG})
	if err != nil {
		resp.Body.Close()
		return nil, err
	}
	s := &httpSource{
		name:    "mjpeg " + Redact(clean) + " via " + dec.Backend().Name,
		resp:    resp,
		decoder: dec,
		ready:   make(chan struct{}, 1),
		done:    make(chan struct{}),
	}
	go s.read(params["boundary"])
	return s, nil
}

func (s *httpSource) read(boundary string) {
	defer close(s.done)
	mr := multipart.NewReader(bufio.NewReaderSize(s.resp.Body, 256*1024), boundary)
	for {
		part, err := mr.NextPart()
		if err != nil {
			s.err = err
			return
		}
		s.mu.Lock()
		p := s.spare
		s.spare = nil
		s.mu.Unlock()
		if p == nil {
			p = &jpegPart{}
		}
		p.data.Reset()
		_, err = p.data.ReadFrom(part)
		part.Close()
		if err != nil {
			s.err = err
			return
		}
		p.arrived = time.Now()
		s.mu.Lock()
		if s.pending != nil {
			s.spare = s.pending // never read: the decoder is behind
		}
		s.pending = p
		s.mu.Unlock()
		select {
		case s.ready <- struct{}{}:
		default:
		}
	}
}

func (s *httpSource) Name() string        { return s.name }
func (s *httpSource) Matrix() ColorMatrix { return MatrixBT601 }

func (s *httpSource) Next(timeout time.Duration) (*gaal.VideoFrame, FrameTiming, error) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		s.mu.Lock()
		p := s.pending
		s.pending = nil
		s.mu.Unlock()
		if p == nil {
			select {
			case <-s.ready:
				continue
			case <-s.done:
				if s.err != nil && !errors.Is(s.err, io.EOF) {
					return nil, FrameTiming{}, RedactError(fmt.Errorf("stream: %w", s.err))
				}
				return nil, FrameTiming{}, io.EOF
			case <-timer.C:
				return nil, FrameTiming{}, ErrTimeout
			}
		}
		t := FrameTiming{Captured: p.arrived, Received: p.arrived, Sent: time.Now()}
		err := s.decoder.Send(gaal.Packet{Data: p.data.Bytes(), Keyframe: true})
		var f *gaal.VideoFrame
		if err == nil {
			f, err = gaal.ReceiveWait(s.decoder, timeout)
		}
		s.mu.Lock()
		if s.spare == nil {
			s.spare = p
		}
		s.mu.Unlock()
		if errors.Is(err, gaal.ErrCorrupt) || errors.Is(err, gaal.ErrInvalid) || errors.Is(err, gaal.ErrAgain) || errors.Is(err, gaal.ErrTimeout) {
			continue // skip a broken JPEG
		}
		if err != nil {
			return nil, t, RedactError(err)
		}
		if f.Range == gaal.RangeUnspecified {
			f.Range = gaal.RangeFull
		}
		f.PTS = time.Duration(t.Received.UnixNano())
		t.Decoded = time.Now()
		return f, t, nil
	}
}

func (s *httpSource) Close() error {
	err := s.resp.Body.Close()
	<-s.done
	s.decoder.Close()
	return err
}

// --- adapters between image.Image cameras and frame sources ---

// imageSource adapts a Camera to a VideoSource, converting images to I420.
type imageSource struct {
	cam    Camera
	name   string
	frames gaal.FramePool
}

func newImageSource(cam Camera, name string) *imageSource {
	return &imageSource{cam: cam, name: name}
}

func (s *imageSource) Name() string        { return s.name }
func (s *imageSource) Matrix() ColorMatrix { return MatrixBT601 }
func (s *imageSource) Close() error        { return s.cam.Close() }

func (s *imageSource) Next(timeout time.Duration) (*gaal.VideoFrame, FrameTiming, error) {
	img, err := s.cam.NextFrame()
	if err != nil {
		return nil, FrameTiming{}, err
	}
	now := time.Now()
	f := ImageToFrame(img, &s.frames)
	return f, FrameTiming{Captured: now, Received: now, Sent: now, Decoded: time.Now()}, nil
}

// ImageToFrame converts img to a full-range I420 frame. YCbCr images keep
// their samples.
func ImageToFrame(img image.Image, pool *gaal.FramePool) *gaal.VideoFrame {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	f := pool.Get(w, h, gaal.I420)
	f.Range = gaal.RangeFull
	cw, ch := (w+1)/2, (h+1)/2
	if yc, ok := img.(*image.YCbCr); ok {
		for y := range h {
			o := yc.YOffset(b.Min.X, b.Min.Y+y)
			copy(f.Y[y*f.YStride:y*f.YStride+w], yc.Y[o:])
		}
		for y := range ch {
			ud := f.U[y*f.UVStride : y*f.UVStride+cw]
			vd := f.V[y*f.UVStride : y*f.UVStride+cw]
			if yc.SubsampleRatio == image.YCbCrSubsampleRatio420 {
				o := yc.COffset(b.Min.X, b.Min.Y+2*y)
				copy(ud, yc.Cb[o:])
				copy(vd, yc.Cr[o:])
				continue
			}
			for x := range cw {
				o := yc.COffset(b.Min.X+2*x, b.Min.Y+2*y)
				ud[x], vd[x] = yc.Cb[o], yc.Cr[o]
			}
		}
		return f
	}
	for y := range h {
		for x := range w {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			ri, gi, bi := int32(r>>8), int32(g>>8), int32(bl>>8)
			f.Y[y*f.YStride+x] = uint8((19595*ri + 38470*gi + 7471*bi + 1<<15) >> 16)
			if x&1 == 0 && y&1 == 0 {
				cb := (-11056*ri - 21712*gi + 32768*bi + 1<<15 + 128<<16) >> 16
				cr := (32768*ri - 27440*gi - 5328*bi + 1<<15 + 128<<16) >> 16
				f.U[(y/2)*f.UVStride+x/2] = clamp8(cb)
				f.V[(y/2)*f.UVStride+x/2] = clamp8(cr)
			}
		}
	}
	return f
}

func clamp8(v int32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// sourceCamera adapts a VideoSource to the image.Image Camera interface,
// keeping only the newest frame.
type sourceCamera struct {
	src    VideoSource
	mu     sync.Mutex
	latest *gaal.VideoFrame
	ready  chan struct{}
	done   chan struct{}
	stop   chan struct{}
	err    error
}

func newSourceCamera(src VideoSource) *sourceCamera {
	c := &sourceCamera{src: src, ready: make(chan struct{}, 1), done: make(chan struct{}), stop: make(chan struct{})}
	go c.pump()
	return c
}

func (c *sourceCamera) pump() {
	defer close(c.done)
	for {
		select {
		case <-c.stop:
			return
		default:
		}
		f, _, err := c.src.Next(time.Second)
		if errors.Is(err, ErrTimeout) {
			continue
		}
		if err != nil {
			c.err = err
			return
		}
		c.mu.Lock()
		old := c.latest
		c.latest = f
		c.mu.Unlock()
		if old != nil {
			old.Release()
		}
		select {
		case c.ready <- struct{}{}:
		default:
		}
	}
}

func (c *sourceCamera) NextFrame() (image.Image, error) {
	for {
		c.mu.Lock()
		f := c.latest
		c.latest = nil
		c.mu.Unlock()
		if f != nil {
			defer f.Release()
			return FrameToImage(f), nil
		}
		select {
		case <-c.ready:
		case <-c.done:
			if c.err != nil {
				return nil, c.err
			}
			return nil, io.EOF
		}
	}
}

func (c *sourceCamera) Close() error {
	close(c.stop)
	<-c.done
	c.mu.Lock()
	if c.latest != nil {
		c.latest.Release()
		c.latest = nil
	}
	c.mu.Unlock()
	return c.src.Close()
}

// FrameToImage copies f into a new I420 image.
func FrameToImage(f *gaal.VideoFrame) *image.YCbCr {
	w, h := f.Width, f.Height
	img := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio420)
	for y := range h {
		copy(img.Y[y*img.YStride:y*img.YStride+w], f.Y[y*f.YStride:])
	}
	cw, ch := (w+1)/2, (h+1)/2
	for y := range ch {
		cb := img.Cb[y*img.CStride : y*img.CStride+cw]
		cr := img.Cr[y*img.CStride : y*img.CStride+cw]
		if f.Format == gaal.NV12 {
			s := f.U[y*f.UVStride : y*f.UVStride+2*cw]
			for x := range cw {
				cb[x], cr[x] = s[2*x], s[2*x+1]
			}
		} else {
			copy(cb, f.U[y*f.UVStride:])
			copy(cr, f.V[y*f.UVStride:])
		}
	}
	return img
}
