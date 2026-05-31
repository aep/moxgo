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

	"github.com/aep/moxgo/pkg/onnx"
	"github.com/aep/moxgo/pkg/registry"
	"github.com/aep/moxgo/pkg/server"
	"github.com/spf13/cobra"
)

var (
	serveConfig string
	serveAddr   string
	serveLib    string
)

func init() {
	serveCmd.Flags().StringVarP(&serveConfig, "config", "c", "moxgo.yaml", "model registry config file")
	serveCmd.Flags().StringVar(&serveAddr, "addr", ":8080", "listen address")
	serveCmd.Flags().StringVar(&serveLib, "lib", "", "path to libonnxruntime.so (auto-detect if empty)")
	rootCmd.AddCommand(serveCmd)
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the inference server (gRPC + HTTP/JSON)",
	RunE:  runServe,
}

func runServe(cmd *cobra.Command, args []string) error {
	var rt *onnx.Runtime
	var err error
	if serveLib != "" {
		rt, err = onnx.LoadFrom(serveLib)
	} else {
		rt, err = onnx.Load()
	}
	if err != nil {
		return fmt.Errorf("load ONNX Runtime: %w", err)
	}
	defer rt.Close()

	// Print ORT info
	fmt.Printf("ONNX Runtime %s (api %d)\n", rt.VersionString(), rt.APIVersion)
	providers, _ := rt.GetAvailableProviders()
	short := make([]string, len(providers))
	for i, p := range providers {
		short[i] = strings.TrimSuffix(p, "ExecutionProvider")
	}
	fmt.Printf("  Providers: %s\n", strings.Join(short, ", "))

	// Load registry from config file (if it exists)
	var reg *server.Registry
	if _, err := os.Stat(serveConfig); err == nil {
		reg, err = server.LoadRegistry(serveConfig)
		if err != nil {
			return err
		}
	} else {
		reg = server.NewRegistry()
	}

	// Auto-discover pulled models from ~/.moxgo/models/
	if localModels, err := registry.ListLocal(); err == nil {
		for _, m := range localModels {
			if err := reg.Add(m); err != nil {
				fmt.Fprintf(os.Stderr, "  warning: %v\n", err)
			}
		}
	}

	models := reg.List()
	fmt.Printf("  Models:    %d\n", len(models))
	for _, m := range models {
		fmt.Printf("    %-20s %s\n", m.Name, m.Path)
	}

	// Create server
	pull := func(name string, progress func(string, int64, int64)) (*server.ModelConfig, error) {
		cb := func(p registry.Progress) {
			if progress != nil {
				progress(p.File, p.Total, p.Downloaded)
			}
		}
		if _, err := registry.Pull(registry.DefaultRegistry, name, cb); err != nil {
			return nil, err
		}
		models, err := registry.ListLocal()
		if err != nil {
			return nil, err
		}
		for _, m := range models {
			if m.Name == name {
				return m, nil
			}
		}
		return nil, fmt.Errorf("model %q not found after pull", name)
	}
	srv := server.New(rt, reg, pull)
	defer srv.Shutdown()

	path, handler := srv.Handler()
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
		next.ServeHTTP(sw, r)
		// Extract RPC method from path (last component)
		method := r.URL.Path
		if i := strings.LastIndex(method, "/"); i >= 0 {
			method = method[i+1:]
		}
		logger.Printf("%s %s %d %v", r.RemoteAddr, method, sw.status, time.Since(start))
	})
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
