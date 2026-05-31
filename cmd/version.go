package cmd

import (
	"context"
	"fmt"

	"connectrpc.com/connect"

	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/aep/moxgo/pkg/version"
	"github.com/spf13/cobra"
)

func init() {
	rootCmd.AddCommand(versionCmd)
}

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print client and server version",
	RunE: func(cmd *cobra.Command, args []string) error {
		fmt.Printf("client: %s\n", version.Version)

		client := newClient()
		resp, err := client.Info(context.Background(), connect.NewRequest(&gomaxv1.InfoRequest{}))
		if err != nil {
			fmt.Printf("server: unavailable\n")
			return nil
		}
		fmt.Printf("server: %s\n", resp.Msg.Version)
		return nil
	},
}
