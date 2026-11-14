package cmd

import (
	"context"
	"net/http"

	"connectrpc.com/connect"

	"github.com/aep/moxgo/pkg/onnx"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/aep/moxgo/pkg/proto/gomax/v1/gomaxv1connect"
	"github.com/aep/moxgo/pkg/server"
	"github.com/spf13/cobra"
)

var (
	serverAddr string
	configPath string
	libPath    string

	// localSrv/localRT hold the in-process session, lazily created by newClient
	// when no remote --server is configured. They are torn down in PersistentPostRun.
	localSrv *server.Server
	localRT  *onnx.Runtime
)

var rootCmd = &cobra.Command{
	Use:           "moxgo",
	Short:         "moxgo inference engine",
	Long:          "ONNX model inference server and client.",
	SilenceUsage:  true,
	SilenceErrors: true,
	PersistentPostRun: func(_ *cobra.Command, _ []string) {
		if localSrv != nil {
			localSrv.Shutdown()
		}
		if localRT != nil {
			localRT.Close()
		}
	},
}

func init() {
	rootCmd.PersistentFlags().StringVar(&serverAddr, "server", "", "remote server address (default: run in-process)")
	rootCmd.PersistentFlags().StringVarP(&configPath, "config", "c", "moxgo.yaml", "model registry config file")
	rootCmd.PersistentFlags().StringVar(&libPath, "lib", "", "path to libonnxruntime.so (auto-detect if empty)")
}

func Execute() error {
	return rootCmd.Execute()
}

// backend is the subset of operations the CLI needs. It is satisfied both by a
// remote HTTP client and by an in-process session, so the command code is
// identical regardless of where inference runs.
//
// The unary methods match *server.Server exactly; the streaming methods are
// adapted so both transports expose a common eventStream.
type backend interface {
	Info(context.Context, *connect.Request[gomaxv1.InfoRequest]) (*connect.Response[gomaxv1.InfoResponse], error)
	ListProviders(context.Context, *connect.Request[gomaxv1.ListProvidersRequest]) (*connect.Response[gomaxv1.ListProvidersResponse], error)
	List(context.Context, *connect.Request[gomaxv1.ListRequest]) (*connect.Response[gomaxv1.ListResponse], error)
	Ps(context.Context, *connect.Request[gomaxv1.PsRequest]) (*connect.Response[gomaxv1.PsResponse], error)
	Run(context.Context, *connect.Request[gomaxv1.RunRequest]) (*connect.Response[gomaxv1.RunResponse], error)
	Rm(context.Context, *connect.Request[gomaxv1.RmRequest]) (*connect.Response[gomaxv1.RmResponse], error)
	Predict(context.Context, *connect.Request[gomaxv1.PredictRequest]) (*connect.Response[gomaxv1.PredictResponse], error)
	Pull(context.Context, *connect.Request[gomaxv1.PullRequest]) (eventStream[gomaxv1.PullEvent], error)
	PredictStream(context.Context, *connect.Request[gomaxv1.PredictStreamRequest]) (eventStream[gomaxv1.PredictStreamResponse], error)
}

// eventStream is the common server-stream shape. *connect.ServerStreamForClient
// satisfies it directly, and chanStream provides it for in-process sessions.
type eventStream[T any] interface {
	Receive() bool
	Msg() *T
	Err() error
}

// newClient returns a backend. With --server set it talks to a remote server
// over HTTP; otherwise it lazily builds an in-process session.
func newClient() (backend, error) {
	if serverAddr != "" {
		return httpBackend{gomaxv1connect.NewInferenceServiceClient(http.DefaultClient, serverAddr)}, nil
	}
	if localSrv == nil {
		srv, rt, err := newLocalServer()
		if err != nil {
			return nil, err
		}
		localSrv, localRT = srv, rt
	}
	return localBackend{localSrv}, nil
}

// --- remote backend ---

type httpBackend struct {
	gomaxv1connect.InferenceServiceClient
}

func (h httpBackend) Pull(ctx context.Context, req *connect.Request[gomaxv1.PullRequest]) (eventStream[gomaxv1.PullEvent], error) {
	return h.InferenceServiceClient.Pull(ctx, req)
}

func (h httpBackend) PredictStream(ctx context.Context, req *connect.Request[gomaxv1.PredictStreamRequest]) (eventStream[gomaxv1.PredictStreamResponse], error) {
	return h.InferenceServiceClient.PredictStream(ctx, req)
}

// --- local in-process backend ---

type localBackend struct {
	*server.Server
}

func (l localBackend) Pull(_ context.Context, req *connect.Request[gomaxv1.PullRequest]) (eventStream[gomaxv1.PullEvent], error) {
	return startStream(func(send func(*gomaxv1.PullEvent)) error {
		if err := l.PullModel(req.Msg.Model, func(file string, total, downloaded int64) {
			send(&gomaxv1.PullEvent{File: file, Total: total, Downloaded: downloaded})
		}); err != nil {
			return err
		}
		send(&gomaxv1.PullEvent{Done: true})
		return nil
	}), nil
}

func (l localBackend) PredictStream(_ context.Context, req *connect.Request[gomaxv1.PredictStreamRequest]) (eventStream[gomaxv1.PredictStreamResponse], error) {
	inst, err := l.Manager().GetOrRun(req.Msg.Model)
	if err != nil {
		return nil, err
	}
	return startStream(func(send func(*gomaxv1.PredictStreamResponse)) error {
		return inst.PredictChunked(req.Msg.Inputs, func(r *gomaxv1.PredictStreamResponse) error {
			send(r)
			return nil
		})
	}), nil
}

// --- in-process stream bridge ---

// chanStream adapts a producer goroutine to the eventStream interface.
type chanStream[T any] struct {
	ch   <-chan *T
	errc <-chan error
	cur  *T
	err  error
	done bool
}

func startStream[T any](produce func(send func(*T)) error) *chanStream[T] {
	ch := make(chan *T)
	errc := make(chan error, 1)
	go func() {
		err := produce(func(v *T) { ch <- v })
		errc <- err
		close(ch)
	}()
	return &chanStream[T]{ch: ch, errc: errc}
}

func (s *chanStream[T]) Receive() bool {
	if s.done {
		return false
	}
	msg, ok := <-s.ch
	if !ok {
		s.done = true
		s.err = <-s.errc
		return false
	}
	s.cur = msg
	return true
}

func (s *chanStream[T]) Msg() *T { return s.cur }

func (s *chanStream[T]) Err() error { return s.err }

func newInput(name string, data []byte) []*gomaxv1.Input {
	return []*gomaxv1.Input{{
		Name: name,
		Data: &gomaxv1.Input_FileBytes{FileBytes: data},
	}}
}

// withPrompt adds a "prompt" text input (for sam3 pipelines) if prompt is set.
func withPrompt(in []*gomaxv1.Input, prompt string) []*gomaxv1.Input {
	if prompt == "" {
		return in
	}
	if len(in) == 1 && in[0].Name == "" {
		in[0].Name = "image"
	}
	return append(in, &gomaxv1.Input{
		Name: "prompt",
		Data: &gomaxv1.Input_FileBytes{FileBytes: []byte(prompt)},
	})
}
