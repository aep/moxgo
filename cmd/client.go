package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"os"
	"os/signal"
	"syscall"
	"time"

	"connectrpc.com/connect"

	"github.com/aep/moxgo/pkg/camera"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/spf13/cobra"
)

// --- run <model> [-f file] ---

var (
	runFile      string
	runInputName string
	runJSON      bool
	runEP        string
	runCamera    bool
	runCameraID  int
	runFPS       float64
	runStream    string
	runPrompt    string
)

func init() {
	runCmd.Flags().StringVarP(&runFile, "file", "f", "", "input file for inference")
	runCmd.Flags().StringVarP(&runInputName, "input", "n", "", "input name (optional if single input)")
	runCmd.Flags().BoolVar(&runJSON, "json", false, "output raw JSON")
	runCmd.Flags().StringVar(&runEP, "ep", "", "execution provider (e.g. CUDA, CoreML)")
	runCmd.Flags().BoolVar(&runCamera, "camera", false, "capture from camera instead of file")
	runCmd.Flags().IntVar(&runCameraID, "camera-id", 0, "camera device index")
	runCmd.Flags().Float64Var(&runFPS, "fps", 1, "frames per second for camera/stream capture")
	runCmd.Flags().StringVar(&runStream, "stream", "", "video stream URL (http:// for MJPEG, rtsp:// for RTSP)")
	runCmd.Flags().StringVar(&runPrompt, "prompt", "", "comma-separated text prompts (sam3 models)")
	rootCmd.AddCommand(runCmd)
}

var runCmd = &cobra.Command{
	Use:   "run <model>",
	Short: "Load a model and optionally run inference",
	Args:  cobra.ExactArgs(1),
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
		if nSources > 1 {
			return fmt.Errorf("--file, --camera, and --stream are mutually exclusive")
		}

		c, err := newClient()
		if err != nil {
			return err
		}
		model := args[0]

		resp, err := c.Run(context.Background(),
			connect.NewRequest(&gomaxv1.RunRequest{Model: model, Ep: runEP}))
		if err != nil {
			return err
		}
		fmt.Printf("%s (ep=%s)\n", resp.Msg.Model, resp.Msg.Ep)

		if runCamera {
			return runCameraLoop(c, model)
		}
		if runStream != "" {
			return runStreamLoop(c, model)
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
			fmt.Printf("    %-20s %.1f%%  [%.0f, %.0f, %.0f, %.0f]\n",
				d.Label, d.Confidence*100, d.X1, d.Y1, d.X2, d.Y2)
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
	Short: "List available cameras",
	RunE: func(cmd *cobra.Command, args []string) error {
		devices, err := camera.List()
		if err != nil {
			return err
		}
		if len(devices) == 0 {
			fmt.Println("No cameras found.")
			return nil
		}
		fmt.Printf("%-5s %s\n", "ID", "NAME")
		for _, d := range devices {
			fmt.Printf("%-5d %s\n", d.Index, d.Name)
		}
		return nil
	},
}

// --- capture loops ---

func runVideoLoop(c backend, model string, cam camera.Camera) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	ticker := time.NewTicker(time.Duration(float64(time.Second) / runFPS))
	defer ticker.Stop()

	frame := 0
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nStopped.")
			return nil
		case <-ticker.C:
			img, err := cam.NextFrame()
			if err != nil {
				fmt.Fprintf(os.Stderr, "frame error: %v\n", err)
				continue
			}

			var buf bytes.Buffer
			if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}); err != nil {
				fmt.Fprintf(os.Stderr, "encode error: %v\n", err)
				continue
			}

			pred, err := c.Predict(context.Background(),
				connect.NewRequest(&gomaxv1.PredictRequest{
					Model:  model,
					Inputs: withPrompt(newInput(runInputName, buf.Bytes()), runPrompt),
				}))
			if err != nil {
				fmt.Fprintf(os.Stderr, "predict error: %v\n", err)
				continue
			}

			frame++
			fmt.Printf("\n--- Frame %d ---\n", frame)
			if runJSON {
				printJSON(pred.Msg)
			} else {
				printPredictResponse(pred.Msg)
			}
		}
	}
}

func runStreamLoop(c backend, model string) error {
	fmt.Printf("Stream: %s\n", runStream)
	cam, err := camera.OpenStream(runStream)
	if err != nil {
		return err
	}
	defer cam.Close()
	fmt.Printf("Capturing at %.1f fps (Ctrl+C to stop)\n", runFPS)
	return runVideoLoop(c, model, cam)
}

func runCameraLoop(c backend, model string) error {
	devices, _ := camera.List()
	if runCameraID < len(devices) {
		fmt.Printf("Camera: [%d] %s\n", devices[runCameraID].Index, devices[runCameraID].Name)
	}

	cam, err := camera.Open(runCameraID)
	if err != nil {
		return err
	}
	defer cam.Close()

	fmt.Printf("Capturing at %.1f fps (Ctrl+C to stop)\n", runFPS)
	return runVideoLoop(c, model, cam)
}
