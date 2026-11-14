package cmd

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"connectrpc.com/connect"
	"github.com/spf13/cobra"
)

var serveAddr string

func init() {
	serveCmd.Flags().StringVar(&serveAddr, "addr", ":8080", "listen address")
	rootCmd.AddCommand(serveCmd)
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the inference server (gRPC + HTTP/JSON)",
	RunE:  runServe,
}

func runServe(cmd *cobra.Command, args []string) error {
	srv, rt, err := newLocalServer()
	if err != nil {
		return err
	}
	defer rt.Close()
	defer srv.Shutdown()

	// Print ORT info
	fmt.Printf("ONNX Runtime %s (api %d)\n", rt.VersionString(), rt.APIVersion)
	providers, _ := rt.GetAvailableProviders()
	short := make([]string, len(providers))
	for i, p := range providers {
		short[i] = strings.TrimSuffix(p, "ExecutionProvider")
	}
	fmt.Printf("  Providers: %s\n", strings.Join(short, ", "))

	models := srv.Models()
	fmt.Printf("  Models:    %d\n", len(models))
	for _, m := range models {
		fmt.Printf("    %-20s %s\n", m.Name, m.Path)
	}

	path, handler := srv.Handler(connect.WithInterceptors(errorRecorder{}))
	mux := http.NewServeMux()
	mux.Handle(path, logRequests(handler))

	httpServer := &http.Server{
		Addr:      serveAddr,
		Handler:   mux,
		Protocols: &http.Protocols{},
	}
	httpServer.Protocols.SetHTTP1(true)
	httpServer.Protocols.SetHTTP2(true)
	httpServer.Protocols.SetUnencryptedHTTP2(true)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		fmt.Printf("Listening on %s\n", serveAddr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "server error: %v\n", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	fmt.Println("\nShutting down...")

	shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return httpServer.Shutdown(shutCtx)
}

func logRequests(next http.Handler) http.Handler {
	logger := log.New(os.Stderr, "", log.LstdFlags)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}
		rpcErr := new(error)
		next.ServeHTTP(sw, r.WithContext(context.WithValue(r.Context(), rpcErrKey{}, rpcErr)))
		// Extract RPC method from path (last component)
		method := r.URL.Path
		if i := strings.LastIndex(method, "/"); i >= 0 {
			method = method[i+1:]
		}
		if *rpcErr != nil {
			logger.Printf("%s %s %d %v: %v", r.RemoteAddr, method, sw.status, time.Since(start), *rpcErr)
			return
		}
		logger.Printf("%s %s %d %v", r.RemoteAddr, method, sw.status, time.Since(start))
	})
}

// rpcErrKey carries a *error from logRequests to errorRecorder, which fills it
// with the RPC's error so the access log can show why a call failed. (gRPC
// reports errors in trailers with HTTP 200, so the status alone can't tell.)
type rpcErrKey struct{}

type errorRecorder struct{}

func (errorRecorder) record(ctx context.Context, err error) error {
	if p, ok := ctx.Value(rpcErrKey{}).(*error); ok && err != nil {
		*p = err
	}
	return err
}

func (e errorRecorder) WrapUnary(next connect.UnaryFunc) connect.UnaryFunc {
	return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
		resp, err := next(ctx, req)
		return resp, e.record(ctx, err)
	}
}

func (errorRecorder) WrapStreamingClient(next connect.StreamingClientFunc) connect.StreamingClientFunc {
	return next
}

func (e errorRecorder) WrapStreamingHandler(next connect.StreamingHandlerFunc) connect.StreamingHandlerFunc {
	return func(ctx context.Context, conn connect.StreamingHandlerConn) error {
		return e.record(ctx, next(ctx, conn))
	}
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
