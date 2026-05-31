package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"connectrpc.com/connect"

	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/spf13/cobra"
)

// --- run <model> [-f file] ---

var (
	runFile      string
	runInputName string
	runJSON      bool
	runEP        string
)

func init() {
	runCmd.Flags().StringVarP(&runFile, "file", "f", "", "input file for inference")
	runCmd.Flags().StringVarP(&runInputName, "input", "n", "", "input name (optional if single input)")
	runCmd.Flags().BoolVar(&runJSON, "json", false, "output raw JSON")
	runCmd.Flags().StringVar(&runEP, "ep", "", "execution provider (e.g. CUDA, CoreML)")
	rootCmd.AddCommand(runCmd)
}

var runCmd = &cobra.Command{
	Use:   "run <model>",
	Short: "Load a model and optionally run inference",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		c := newClient()
		model := args[0]

		resp, err := c.Run(context.Background(),
			connect.NewRequest(&gomaxv1.RunRequest{Model: model, Ep: runEP}))
		if err != nil {
			return err
		}
		fmt.Printf("%s (ep=%s)\n", resp.Msg.Model, resp.Msg.Ep)

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
				Inputs: newInput(runInputName, data),
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
		resp, err := newClient().Ps(context.Background(),
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
		resp, err := newClient().List(context.Background(),
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
		_, err := newClient().Rm(context.Background(),
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
)

func init() {
	predictCmd.Flags().StringVarP(&predictFile, "file", "f", "", "input file (required)")
	predictCmd.Flags().StringVarP(&predictInputName, "input", "n", "", "input name (optional if single input)")
	predictCmd.Flags().BoolVar(&predictJSON, "json", false, "output raw JSON")
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
		resp, err := newClient().Predict(context.Background(),
			connect.NewRequest(&gomaxv1.PredictRequest{
				Model:  args[0],
				Inputs: newInput(predictInputName, data),
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
		stream, err := newClient().PredictStream(context.Background(),
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
		resp, err := newClient().ListProviders(context.Background(),
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
