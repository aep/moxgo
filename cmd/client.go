package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/bits"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"connectrpc.com/connect"
	"github.com/aep/gaal"

	"github.com/aep/moxgo/pkg/camera"
	"github.com/aep/moxgo/pkg/live"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// --- run <model> [-f file] ---

var (
	runFile      string
	runInputName string
	runJSON      bool
	runEP        string
	runCamera    bool
	runCameraID  int
	runCameraRes string
	runFPS       float64
	runStream    string
	runPrompt    string
)

func init() {
	runCmd.Flags().StringVarP(&runFile, "file", "f", "", "input file for inference")
	runCmd.Flags().StringVarP(&runInputName, "input", "n", "", "input name (optional if single input)")
	runCmd.Flags().BoolVar(&runJSON, "json", false, "output raw JSON")
	runCmd.Flags().StringVar(&runEP, "ep", "", "execution provider (e.g. CUDA, CoreML)")
	cam := runCmd.Flags().VarPF(cameraFlag{}, "camera", "", "capture from a camera instead of a file (--camera N picks device N)")
	cam.NoOptDefVal = "true"
	runCmd.Flags().IntVar(&runCameraID, "camera-id", 0, "camera device index")
	runCmd.Flags().StringVar(&runCameraRes, "camera-res", "640x480", "requested camera resolution WxH (driver picks the closest it supports)")
	runCmd.Flags().Float64Var(&runFPS, "fps", 1, "frames per second for camera/stream capture")
	runCmd.Flags().StringVar(&runStream, "stream", "", "video stream URL (http:// for MJPEG, rtsp:// for RTSP)")
	runCmd.Flags().StringVar(&runPrompt, "prompt", "", "comma-separated text prompts (sam3 models)")
	rootCmd.AddCommand(runCmd)
}

var runCmd = &cobra.Command{
	Use:   "run <model>",
	Short: "Load a model and optionally run inference",
	Long: `Load a model and optionally run inference on a file, a camera or a stream.

With --display or --stats it runs live: video goes from the source to the
window at the source's rate, detection works on the newest frame beside it, and
its latest boxes, labels and masks are drawn into the video. Examples:

  moxgo run yolo11n --camera 0 --camera-res 1280x720 --display
  moxgo run yolo11n --stream rtsp://host/stream --display --stats
  moxgo run yolo11n --onvif http://host:8000/onvif/device_service --display

Camera credentials come from the URL or from the MOXGO_CAMERA_USER and
MOXGO_CAMERA_PASSWORD environment variables. They are never printed.`,
	Args: runArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		nSources := 0
		if runFile != "" {
			nSources++
		}
		if runCamera {
			nSources++
		}
		if runStream != "" {
			nSources++
		}
		if runONVIF != "" {
			nSources++
		}
		if nSources > 1 {
			return fmt.Errorf("--file, --camera, --stream and --onvif are mutually exclusive")
		}
		runFPSSet = cmd.Flags().Changed("fps")
		isLive := runDisplay || runStats
		if isLive && (runFile != "" || nSources == 0) {
			return fmt.Errorf("--display and --stats need --camera, --stream or --onvif")
		}

		model, err := parseRunArgs(args)
		if err != nil {
			return err
		}
		c, err := newClient()
		if err != nil {
			return err
		}

		resp, err := c.Run(context.Background(),
			connect.NewRequest(&gomaxv1.RunRequest{Model: model, Ep: runEP}))
		if err != nil {
			return err
		}
		fmt.Printf("%s (ep=%s)\n", resp.Msg.Model, resp.Msg.Ep)

		if isLive {
			return runLive(c, model)
		}
		if runCamera || runStream != "" || runONVIF != "" {
			return runCaptureLoop(c, model)
		}

		if runFile == "" {
			return nil
		}

		data, err := os.ReadFile(runFile)
		if err != nil {
			return err
		}
		pred, err := c.Predict(context.Background(),
			connect.NewRequest(&gomaxv1.PredictRequest{
				Model:  model,
				Inputs: withPrompt(newInput(runInputName, data), runPrompt),
			}))
		if err != nil {
			return err
		}
		if runJSON {
			return printJSON(pred.Msg)
		}
		return printPredictResponse(pred.Msg)
	},
}

// --- ps ---

func init() { rootCmd.AddCommand(psCmd) }

var psCmd = &cobra.Command{
	Use:   "ps",
	Short: "List running models",
	RunE: func(cmd *cobra.Command, args []string) error {
		c, err := newClient()
		if err != nil {
			return err
		}
		resp, err := c.Ps(context.Background(),
			connect.NewRequest(&gomaxv1.PsRequest{}))
		if err != nil {
			return err
		}
		if len(resp.Msg.Models) == 0 {
			fmt.Println("No running models.")
			return nil
		}
		fmt.Printf("%-20s %-8s %s\n", "NAME", "EP", "THREADS")
		for _, m := range resp.Msg.Models {
			fmt.Printf("%-20s %-8s %d\n", m.Name, m.Ep, m.Threads)
		}
		return nil
	},
}

// --- ls ---

func init() { rootCmd.AddCommand(lsCmd) }

var lsCmd = &cobra.Command{
	Use:   "ls",
	Short: "List available models",
	RunE: func(cmd *cobra.Command, args []string) error {
		c, err := newClient()
		if err != nil {
			return err
		}
		resp, err := c.List(context.Background(),
			connect.NewRequest(&gomaxv1.ListRequest{}))
		if err != nil {
			return err
		}
		fmt.Printf("%-20s %-8s %s\n", "NAME", "TYPE", "PATH")
		for _, m := range resp.Msg.Models {
			t := ""
			if len(m.Inputs) > 0 {
				t = m.Inputs[0].Type
			}
			fmt.Printf("%-20s %-8s %s\n", m.Name, t, m.Path)
		}
		return nil
	},
}

// --- rm <model> ---

func init() { rootCmd.AddCommand(rmCmd) }

var rmCmd = &cobra.Command{
	Use:   "rm <model>",
	Short: "Unload a running model",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		c, err := newClient()
		if err != nil {
			return err
		}
		_, err = c.Rm(context.Background(),
			connect.NewRequest(&gomaxv1.RmRequest{Model: args[0]}))
		if err != nil {
			return err
		}
		fmt.Println(args[0])
		return nil
	},
}

// --- predict <model> -f <file> ---

var (
	predictFile      string
	predictInputName string
	predictJSON      bool
	predictPrompt    string
)

func init() {
	predictCmd.Flags().StringVarP(&predictFile, "file", "f", "", "input file (required)")
	predictCmd.Flags().StringVarP(&predictInputName, "input", "n", "", "input name (optional if single input)")
	predictCmd.Flags().BoolVar(&predictJSON, "json", false, "output raw JSON")
	predictCmd.Flags().StringVar(&predictPrompt, "prompt", "", "comma-separated text prompts (sam3 models)")
	_ = predictCmd.MarkFlagRequired("file")
	rootCmd.AddCommand(predictCmd)
}

var predictCmd = &cobra.Command{
	Use:   "predict <model>",
	Short: "Run inference (auto-loads model if needed)",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		data, err := os.ReadFile(predictFile)
		if err != nil {
			return err
		}
		c, err := newClient()
		if err != nil {
			return err
		}
		resp, err := c.Predict(context.Background(),
			connect.NewRequest(&gomaxv1.PredictRequest{
				Model:  args[0],
				Inputs: withPrompt(newInput(predictInputName, data), predictPrompt),
			}))
		if err != nil {
			return err
		}
		if predictJSON {
			return printJSON(resp.Msg)
		}
		return printPredictResponse(resp.Msg)
	},
}

// --- stream <model> -f <file> ---

var (
	streamFile      string
	streamInputName string
)

func init() {
	streamCmd.Flags().StringVarP(&streamFile, "file", "f", "", "input audio file (required)")
	streamCmd.Flags().StringVarP(&streamInputName, "input", "n", "", "input name")
	_ = streamCmd.MarkFlagRequired("file")
	rootCmd.AddCommand(streamCmd)
}

var streamCmd = &cobra.Command{
	Use:   "stream <model>",
	Short: "Run chunked audio inference",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		data, err := os.ReadFile(streamFile)
		if err != nil {
			return err
		}
		c, err := newClient()
		if err != nil {
			return err
		}
		stream, err := c.PredictStream(context.Background(),
			connect.NewRequest(&gomaxv1.PredictStreamRequest{
				Model:  args[0],
				Inputs: newInput(streamInputName, data),
			}))
		if err != nil {
			return err
		}
		for stream.Receive() {
			msg := stream.Msg()
			fmt.Printf("[%.1fs - %.1fs]\n", msg.WindowStartSec, msg.WindowEndSec)
			for _, out := range msg.Outputs {
				printOutputResult(out)
			}
		}
		return stream.Err()
	},
}

// --- providers ---

func init() { rootCmd.AddCommand(providersCmd) }

var providersCmd = &cobra.Command{
	Use:   "providers",
	Short: "List execution providers",
	RunE: func(cmd *cobra.Command, args []string) error {
		c, err := newClient()
		if err != nil {
			return err
		}
		resp, err := c.ListProviders(context.Background(),
			connect.NewRequest(&gomaxv1.ListProvidersRequest{}))
		if err != nil {
			return err
		}
		for _, p := range resp.Msg.Providers {
			fmt.Println(p)
		}
		return nil
	},
}

// --- output helpers ---

func printPredictResponse(resp *gomaxv1.PredictResponse) error {
	fmt.Printf("Inference: %.2fms\n", resp.InferenceTimeMs)
	for _, out := range resp.Outputs {
		printOutputResult(out)
	}
	return nil
}

func printOutputResult(out *gomaxv1.OutputResult) {
	fmt.Printf("  %s %v\n", out.Name, out.Shape)
	switch r := out.Result.(type) {
	case *gomaxv1.OutputResult_Classifications:
		for i, c := range r.Classifications.Items {
			fmt.Printf("    #%d  %-20s  %.1f%%\n", i+1, c.Label, c.Score*100)
		}
	case *gomaxv1.OutputResult_Detections:
		for _, d := range r.Detections.Items {
			mask := ""
			if m := d.Mask; m != nil {
				n := 0
				for _, b := range m.Bits {
					n += bits.OnesCount8(b)
				}
				mask = fmt.Sprintf("  mask=%dx%d (%d px)", m.Width, m.Height, n)
			}
			fmt.Printf("    %-20s %.1f%%  [%.0f, %.0f, %.0f, %.0f]%s\n",
				d.Label, d.Confidence*100, d.X1, d.Y1, d.X2, d.Y2, mask)
		}
	case *gomaxv1.OutputResult_Embedding:
		fmt.Printf("    embedding: %d values\n", len(r.Embedding.Values))
	case *gomaxv1.OutputResult_Raw:
		fmt.Printf("    raw: %d bytes\n", len(r.Raw.Data))
	}
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

// --- cameras ---

func init() { rootCmd.AddCommand(camerasCmd) }

var camerasCmd = &cobra.Command{
	Use:   "cameras",
	Short: "List available cameras (with --onvif, also network cameras)",
	RunE: func(cmd *cobra.Command, args []string) error {
		devices, err := camera.List()
		if err != nil && !errors.Is(err, camera.ErrNotSupported) {
			return err
		}
		if len(devices) == 0 {
			fmt.Println("No cameras found.")
		} else {
			fmt.Printf("%-5s %s\n", "ID", "NAME")
			for _, d := range devices {
				fmt.Printf("%-5d %s\n", d.Index, d.Name)
			}
		}
		if camerasONVIF {
			return listONVIF(cmd.Context())
		}
		return nil
	},
}

// --- capture loop ---

// cameraFlag is --camera: a switch, or the device number with --camera=N.
type cameraFlag struct{}

func (cameraFlag) String() string   { return strconv.FormatBool(runCamera) }
func (cameraFlag) Type() string     { return "bool" }
func (cameraFlag) IsBoolFlag() bool { return true }
func (cameraFlag) Set(s string) error {
	if v, err := strconv.ParseBool(s); err == nil && s != "0" && s != "1" {
		runCamera = v
		return nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return fmt.Errorf("want a camera number, got %q", s)
	}
	runCamera, runCameraID = true, n
	return nil
}

var _ pflag.Value = cameraFlag{}

// runArgs accepts <model>, and with --camera also a device number next to it.
func runArgs(cmd *cobra.Command, args []string) error {
	if len(args) == 2 && runCamera {
		return nil
	}
	return cobra.ExactArgs(1)(cmd, args)
}

func parseRunArgs(args []string) (string, error) {
	if len(args) == 1 {
		return args[0], nil
	}
	if n, err := strconv.Atoi(args[1]); err == nil && n >= 0 {
		runCameraID = n
		return args[0], nil
	}
	if n, err := strconv.Atoi(args[0]); err == nil && n >= 0 {
		runCameraID = n
		return args[1], nil
	}
	return "", fmt.Errorf("expected a model and a camera number, got %q and %q", args[0], args[1])
}

// runCaptureLoop infers on the newest frame of the source at --fps, printing
// results. Frames go out as YCbCr; nothing is encoded.
func runCaptureLoop(c backend, model string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	src, err := openLiveSource(ctx)
	if err != nil {
		return err
	}
	defer src.Close()
	fmt.Printf("Source: %s\n", camera.Redact(src.Name()))
	fmt.Printf("Capturing at %.1f fps (Ctrl+C to stop)\n", runFPS)

	frames := live.NewMailbox[*gaal.VideoFrame]()
	pumpDone := make(chan error, 1)
	go func() {
		for ctx.Err() == nil {
			f, _, err := src.Next(time.Second)
			if errors.Is(err, camera.ErrTimeout) {
				continue
			}
			if err != nil {
				pumpDone <- err
				return
			}
			if old, ok := frames.Put(f); ok {
				old.Release()
			}
		}
		pumpDone <- nil
	}()
	defer func() {
		if f, ok := frames.Close(); ok {
			f.Release()
		}
	}()

	det := live.NewRemoteDetector(nil, model, runInputName, runPrompt)
	ticker := time.NewTicker(time.Duration(float64(time.Second) / runFPS))
	defer ticker.Stop()

	frame := 0
	process := func() {
		f, ok := takeNewest(frames)
		if !ok {
			return
		}
		err := det.Prepare(f)
		f.Release()
		if err != nil {
			fmt.Fprintf(os.Stderr, "frame error: %v\n", err)
			return
		}
		pred, err := c.Predict(ctx, connect.NewRequest(det.Request()))
		if err != nil {
			if ctx.Err() == nil {
				fmt.Fprintf(os.Stderr, "predict error: %s\n", camera.Redact(err.Error()))
			}
			return
		}
		frame++
		fmt.Printf("\n--- Frame %d ---\n", frame)
		if runJSON {
			printJSON(pred.Msg)
		} else {
			printPredictResponse(pred.Msg)
		}
	}
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nStopped.")
			return nil
		case err := <-pumpDone:
			process() // the last frame of a finite source
			if err == nil || errors.Is(err, io.EOF) {
				return nil
			}
			return err
		case <-ticker.C:
			process()
		}
	}
}

// takeNewest empties the mailbox without waiting if nothing is there yet.
func takeNewest(m *live.Mailbox[*gaal.VideoFrame]) (*gaal.VideoFrame, bool) {
	return m.TryTake()
}
