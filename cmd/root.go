package cmd

import (
	"net/http"

	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/aep/moxgo/pkg/proto/gomax/v1/gomaxv1connect"
	"github.com/spf13/cobra"
)

var serverAddr string

var rootCmd = &cobra.Command{
	Use:          "moxgo",
	Short:        "moxgo inference engine",
	Long:         "ONNX model inference server and client.",
	SilenceUsage: true,
}

func init() {
	rootCmd.PersistentFlags().StringVar(&serverAddr, "server", "http://localhost:8080", "server address")
}

func Execute() error {
	return rootCmd.Execute()
}

func newClient() gomaxv1connect.InferenceServiceClient {
	return gomaxv1connect.NewInferenceServiceClient(http.DefaultClient, serverAddr)
}

func newInput(name string, data []byte) []*gomaxv1.Input {
	return []*gomaxv1.Input{{
		Name: name,
		Data: &gomaxv1.Input_FileBytes{FileBytes: data},
	}}
}
