package cmd

import (
	"github.com/aep/moxgo/pkg/local"
	"github.com/aep/moxgo/pkg/onnx"
	"github.com/aep/moxgo/pkg/server"
)

// newLocalServer builds an in-process inference session: it loads the ONNX
// runtime, the model registry (config file + locally pulled models), and wires
// up the registry pull. The returned *server.Server is the same backend the
// HTTP server serves, so server-side and local sessions share the same code.
//
// The caller owns the returned runtime and must Close it (and Shutdown the
// server) when done.
func newLocalServer() (*server.Server, *onnx.Runtime, error) {
	rtOpts, err := runtimeOptions()
	if err != nil {
		return nil, nil, err
	}
	return local.New(local.Options{LibPath: libPath, ConfigPath: configPath, Runtime: rtOpts})
}
