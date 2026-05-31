package cmd

import (
	"fmt"
	"strings"

	"github.com/aep/moxgo/pkg/onnx"
	"github.com/spf13/cobra"
)

var inspectLib string

func init() {
	inspectCmd.Flags().StringVar(&inspectLib, "lib", "", "path to libonnxruntime.so (auto-detect if empty)")
	rootCmd.AddCommand(inspectCmd)
}

var inspectCmd = &cobra.Command{
	Use:   "inspect <model.onnx>",
	Short: "Inspect an ONNX model file",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		var rt *onnx.Runtime
		var err error
		if inspectLib != "" {
			rt, err = onnx.LoadFrom(inspectLib)
		} else {
			rt, err = onnx.Load()
		}
		if err != nil {
			return fmt.Errorf("load ONNX Runtime: %w", err)
		}
		defer rt.Close()

		fmt.Printf("ONNX Runtime %s (api %d)\n", rt.VersionString(), rt.APIVersion)
		providers, _ := rt.GetAvailableProviders()
		short := make([]string, len(providers))
		for i, p := range providers {
			short[i] = strings.TrimSuffix(p, "ExecutionProvider")
		}
		fmt.Printf("  Providers: %s\n\n", strings.Join(short, ", "))

		sess, err := rt.OpenSession(args[0])
		if err != nil {
			return fmt.Errorf("open model: %w\n  ORT: %s", err, rt.LastError())
		}
		defer sess.Close()

		fmt.Printf("Model: %s\n", args[0])

		meta, err := sess.Metadata()
		if err == nil && meta != nil {
			if meta.ProducerName != "" {
				fmt.Printf("  Producer:    %s\n", meta.ProducerName)
			}
			if meta.GraphName != "" {
				fmt.Printf("  Graph:       %s\n", meta.GraphName)
			}
			if meta.Domain != "" {
				fmt.Printf("  Domain:      %s\n", meta.Domain)
			}
			if meta.Description != "" {
				fmt.Printf("  Description: %s\n", meta.Description)
			}
			if meta.Version != 0 {
				fmt.Printf("  Version:     %d\n", meta.Version)
			}
			if len(meta.Custom) > 0 {
				fmt.Printf("  Custom metadata:\n")
				for k, v := range meta.Custom {
					if len(v) > 100 {
						v = v[:100] + "..."
					}
					fmt.Printf("    %-30s %s\n", k, v)
				}
			}
		}
		fmt.Println()

		fmt.Printf("Inputs (%d):\n", len(sess.Inputs))
		for _, in := range sess.Inputs {
			fmt.Printf("  %-30s %s %v\n", in.Name, in.Dtype, in.Shape)
		}
		fmt.Println()

		fmt.Printf("Outputs (%d):\n", len(sess.Outputs))
		for _, out := range sess.Outputs {
			fmt.Printf("  %-30s %s %v\n", out.Name, out.Dtype, out.Shape)
		}

		return nil
	},
}
