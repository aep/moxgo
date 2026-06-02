package cmd

import (
	"context"
	"fmt"

	"connectrpc.com/connect"
	"github.com/schollz/progressbar/v3"

	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/aep/moxgo/pkg/registry"
	"github.com/spf13/cobra"
)

var registryURL string

func formatSize(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1fG", float64(b)/float64(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.0fM", float64(b)/float64(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.0fK", float64(b)/float64(1<<10))
	default:
		return fmt.Sprintf("%dB", b)
	}
}

func init() {
	rootCmd.AddCommand(pullCmd)

	searchCmd.Flags().StringVar(&registryURL, "registry", registry.DefaultRegistry, "registry URL")
	rootCmd.AddCommand(searchCmd)
}

var pullCmd = &cobra.Command{
	Use:   "pull <model>",
	Short: "Download a model from the registry",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		name := args[0]
		fmt.Printf("pulling %s\n", name)

		stream, err := newClient().Pull(context.Background(),
			connect.NewRequest(&gomaxv1.PullRequest{Model: name}))
		if err != nil {
			return err
		}

		var bar *progressbar.ProgressBar
		var currentFile string

		for stream.Receive() {
			ev := stream.Msg()
			if ev.Done {
				if bar != nil {
					_ = bar.Finish()
					fmt.Println()
				}
				break
			}

			if ev.File != currentFile {
				if bar != nil {
					_ = bar.Finish()
					fmt.Println()
				}
				currentFile = ev.File
				bar = progressbar.DefaultBytes(ev.Total, "  "+ev.File)
			}
			if bar != nil {
				_ = bar.Set64(ev.Downloaded)
			}
		}
		if err := stream.Err(); err != nil {
			return err
		}
		fmt.Println(name)
		return nil
	},
}

var searchCmd = &cobra.Command{
	Use:   "search",
	Short: "List models available in the registry",
	RunE: func(cmd *cobra.Command, args []string) error {
		idx, err := registry.FetchIndex(registryURL)
		if err != nil {
			return err
		}
		fmt.Printf("%-30s %-10s %-20s %s\n", "NAME", "SIZE", "LICENSE", "DESCRIPTION")
		for _, m := range idx.Models {
			fmt.Printf("%-30s %-10s %-20s %s\n", m.Name, formatSize(m.Size), m.License, m.Description)
		}
		return nil
	},
}
