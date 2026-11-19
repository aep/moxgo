package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"connectrpc.com/connect"
	"github.com/aep/gaal/display"

	"github.com/aep/moxgo/pkg/camera"
	"github.com/aep/moxgo/pkg/live"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
)

var (
	runDisplay        bool
	runStats          bool
	runDuration       time.Duration
	runONVIF          string
	runONVIFProfile   string
	runCameraFPS      int
	runCameraFormat   string
	runDecoder        string
	runRTSPTransport  string
	runKeepDelay      bool
	runDisplayBackend string
	runFullscreen     bool
	runNoDetect       bool
	runFPSSet         bool
	runNoRedraw       bool

	camerasONVIF bool
)

func init() {
	f := runCmd.Flags()
	f.BoolVar(&runDisplay, "display", false, "live mode: show the video with detections in a window")
	f.BoolVar(&runStats, "stats", false, "live mode: print per-stage timings and a summary at the end")
	f.DurationVar(&runDuration, "duration", 0, "stop live mode after this time (default: run until interrupted)")
	f.StringVar(&runONVIF, "onvif", "", "ONVIF camera: device service URL or host[:port]")
	f.StringVar(&runONVIFProfile, "onvif-profile", "", "ONVIF media profile: main, sub, or a name or token (default main)")
	f.IntVar(&runCameraFPS, "camera-fps", 0, "requested camera frame rate (default: the driver's)")
	f.StringVar(&runCameraFormat, "camera-format", "", "force a V4L2 pixel format, e.g. MJPG or YUYV")
	f.StringVar(&runDecoder, "decoder", "", "video decoder backend, e.g. nvdec, vaapi, openh264 (default: best available)")
	f.StringVar(&runRTSPTransport, "rtsp-transport", "tcp", "RTSP transport: tcp, udp or auto")
	f.BoolVar(&runKeepDelay, "rtsp-keep-delay", false, "keep the H.264 decoder's reordering delay (for streams with B-frames)")
	f.StringVar(&runDisplayBackend, "display-backend", "", "display backend: wayland, x11, headless... (default: best available)")
	f.BoolVar(&runFullscreen, "fullscreen", false, "open the window fullscreen")
	f.BoolVar(&runNoDetect, "no-detect", false, "live mode: video only, without inference")
	f.BoolVar(&runNoRedraw, "no-redraw", false, "live mode: do not update a shown frame when its own detections arrive")

	camerasCmd.Flags().BoolVar(&camerasONVIF, "onvif", false, "also discover ONVIF cameras on the network")
}

// sourceConfig is the capture configuration the flags ask for.
func sourceConfig() (camera.SourceConfig, error) {
	cfg := camera.SourceConfig{
		FPS:              runCameraFPS,
		Format:           strings.ToUpper(runCameraFormat),
		Backend:          runDecoder,
		Transport:        runRTSPTransport,
		KeepDecoderDelay: runKeepDelay,
	}
	if runCamera {
		if _, err := fmt.Sscanf(runCameraRes, "%dx%d", &cfg.Width, &cfg.Height); err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
			return cfg, fmt.Errorf("invalid --camera-res %q, want WxH (e.g. 1280x720)", runCameraRes)
		}
	}
	if cfg.Format != "" && len(cfg.Format) != 4 {
		return cfg, fmt.Errorf("invalid --camera-format %q, want a fourcc such as MJPG", runCameraFormat)
	}
	return cfg, nil
}

// openLiveSource opens the source the flags name.
func openLiveSource(ctx context.Context) (camera.VideoSource, error) {
	cfg, err := sourceConfig()
	if err != nil {
		return nil, err
	}
	switch {
	case runCamera:
		devices, _ := camera.List()
		if len(devices) == 0 {
			fmt.Fprintln(os.Stderr, "warning: no usable cameras found (see `moxgo cameras`)")
		}
		return camera.OpenVideoCamera(runCameraID, cfg)
	case runONVIF != "":
		ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
		defer cancel()
		return camera.OpenONVIF(ctx, runONVIF, runONVIFProfile, cfg)
	default:
		fmt.Printf("Stream: %s\n", camera.Redact(runStream))
		return camera.OpenVideoStream(runStream, cfg)
	}
}

// newDetector picks the fastest way to reach the model: in-process on the
// decoder's planes when possible, else through Predict with YCbCr inputs.
func newDetector(c backend, model string, m camera.ColorMatrix) (live.Detector, string, error) {
	if runNoDetect {
		return nil, "none", nil
	}
	if lb, ok := c.(localBackend); ok && runPrompt == "" {
		inst, err := lb.Manager().GetOrRun(model)
		if err != nil {
			return nil, "", err
		}
		d, err := live.NewLocalDetector(inst, m)
		if err == nil {
			return d, "in-process", nil
		}
		fmt.Fprintf(os.Stderr, "note: %v; using the generic path\n", err)
	}
	d := live.NewRemoteDetector(func(ctx context.Context, req *gomaxv1.PredictRequest) (*gomaxv1.PredictResponse, error) {
		resp, err := c.Predict(ctx, connect.NewRequest(req))
		if err != nil {
			return nil, err
		}
		return resp.Msg, nil
	}, model, runInputName, runPrompt)
	// Do not send more pixels than the model can use.
	if list, err := c.List(context.Background(), connect.NewRequest(&gomaxv1.ListRequest{})); err == nil {
		for _, mi := range list.Msg.Models {
			if mi.Name != model || len(mi.Inputs) != 1 {
				continue
			}
			p := mi.Inputs[0].Params
			d.MaxW, _ = strconv.Atoi(p["width"])
			d.MaxH, _ = strconv.Atoi(p["height"])
		}
	}
	return d, "predict (YCbCr)", nil
}

// runLive runs the camera-to-display pipeline.
func runLive(c backend, model string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	src, err := openLiveSource(ctx)
	if err != nil {
		return err
	}
	defer src.Close()
	fmt.Printf("Source: %s\n", camera.Redact(src.Name()))

	det, path, err := newDetector(c, model, src.Matrix())
	if err != nil {
		return err
	}
	if det != nil {
		defer det.Close()
	}
	fmt.Printf("Detector: %s\n", path)

	stats := live.NewStats()
	opts := live.Options{
		Source: src, Detector: det, Stats: stats,
		MaxDetectFPS: 0,
		Duration:     runDuration,
		NoRedraw:     runNoRedraw,
		Log:          func(m string) { fmt.Fprintln(os.Stderr, m) },
	}
	if runFPSSet {
		opts.MaxDetectFPS = runFPS
	}
	if runStats {
		opts.StatsEvery, opts.StatsOut = 2*time.Second, os.Stderr
	}
	if runDisplay {
		d, err := openDisplay(model, src)
		if err != nil {
			return err
		}
		defer d.Close()
		opts.Display = d
	}

	err = live.New(opts).Run(ctx)
	if runStats {
		if d, ok := opts.Display.(*display.Display); ok {
			info := d.Info()
			ds := d.Stats()
			fmt.Printf("Display: %s via %s, presentation latency from %s; %d shown, %d presented, %d replaced before presenting\n",
				info.Backend, info.Path, info.LatencySource, ds.Shown, ds.Presented, ds.Dropped)
		}
		live.WriteSummary(os.Stdout, stats.Summary(), runJSON)
	}
	return err
}

func openDisplay(model string, src camera.VideoSource) (*display.Display, error) {
	opts := display.Options{
		Title:      "moxgo " + model,
		Backend:    runDisplayBackend,
		Fullscreen: runFullscreen,
	}
	switch src.Matrix() {
	case camera.MatrixBT601:
		opts.Matrix = display.BT601
	case camera.MatrixBT709:
		opts.Matrix = display.BT709
	}
	return display.Open(opts)
}

// listONVIF prints the ONVIF devices on the network, with their profiles
// when the credentials in the environment work.
func listONVIF(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	found, err := camera.DiscoverONVIF(ctx, 3*time.Second)
	if err != nil {
		return err
	}
	fmt.Printf("\nONVIF devices: %d\n", len(found))
	for _, d := range found {
		fmt.Printf("  %s  %s (%s)\n", d.Address, d.Name, d.Hardware)
		for _, s := range d.Services {
			fmt.Printf("    service %s\n", s)
			if os.Getenv(camera.EnvUser) == "" {
				continue
			}
			ps, err := camera.ONVIFProfiles(ctx, s, camera.SourceConfig{})
			if err != nil {
				fmt.Printf("    profiles: %s\n", camera.Redact(err.Error()))
				continue
			}
			for _, p := range ps {
				fmt.Printf("    profile %-12s %-8s %dx%d %.0f fps  (token %s)\n", p.Name, p.Codec, p.Width, p.Height, p.FPS, p.Token)
			}
		}
	}
	return nil
}
