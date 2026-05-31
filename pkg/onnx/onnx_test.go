package onnx_test

import (
	"fmt"
	"os"
	"testing"
	"unsafe"

	"github.com/aep/moxgo/pkg/onnx"
)

const modelPath = "../../models/yolo11n.onnx"

func loadTestSession(tb testing.TB) (*onnx.Runtime, *onnx.Session) {
	tb.Helper()
	if _, err := os.Stat(modelPath); err != nil {
		tb.Skipf("model not found: %s", modelPath)
	}
	rt, err := onnx.Load()
	if err != nil {
		tb.Fatalf("Load: %v", err)
	}
	sess, err := rt.OpenSession(modelPath)
	if err != nil {
		rt.Close()
		tb.Fatalf("OpenSession: %v", err)
	}
	return rt, sess
}

// BenchmarkExecLoop benchmarks the steady-state inference loop (Exec + GetOutput).
// It proves zero allocations after the first iteration.
func BenchmarkExecLoop(b *testing.B) {
	rt, sess := loadTestSession(b)
	defer rt.Close()
	defer sess.Close()

	// Pre-allocate input buffer (640x640 RGB float32)
	input := make([]float32, 1*3*640*640)
	for i := range input {
		input[i] = 0.5
	}

	run := sess.NewRun()
	defer run.Close()

	if err := onnx.BindSlice(run, "images", input, []int64{1, 3, 640, 640}); err != nil {
		b.Fatalf("BindSlice: %v", err)
	}

	// Warm up: first call does one-time allocations (prepare)
	if err := run.Exec(); err != nil {
		b.Fatalf("warmup Exec: %v", err)
	}
	if _, err := run.GetOutput(0); err != nil {
		b.Fatalf("warmup GetOutput: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for b.Loop() {
		if err := run.Exec(); err != nil {
			b.Fatalf("Exec: %v", err)
		}
		out, err := run.GetOutput(0)
		if err != nil {
			b.Fatalf("GetOutput: %v", err)
		}
		// Touch output to prevent dead-code elimination
		if out.Len == 0 {
			b.Fatal("empty output")
		}
	}
}

// BenchmarkExecLoopTyped benchmarks with typed OutputSlice (still zero-alloc).
func BenchmarkExecLoopTyped(b *testing.B) {
	rt, sess := loadTestSession(b)
	defer rt.Close()
	defer sess.Close()

	input := make([]float32, 1*3*640*640)
	for i := range input {
		input[i] = 0.5
	}

	run := sess.NewRun()
	defer run.Close()

	if err := onnx.BindSlice(run, "images", input, []int64{1, 3, 640, 640}); err != nil {
		b.Fatalf("BindSlice: %v", err)
	}

	// Warm up
	if err := run.Exec(); err != nil {
		b.Fatalf("warmup Exec: %v", err)
	}
	if _, _, err := onnx.OutputSlice[float32](run, 0); err != nil {
		b.Fatalf("warmup OutputSlice: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for b.Loop() {
		if err := run.Exec(); err != nil {
			b.Fatalf("Exec: %v", err)
		}
		data, shape, err := onnx.OutputSlice[float32](run, 0)
		if err != nil {
			b.Fatalf("OutputSlice: %v", err)
		}
		if len(data) == 0 || len(shape) == 0 {
			b.Fatal("empty output")
		}
	}
}

func TestZeroAllocExec(t *testing.T) {
	rt, sess := loadTestSession(t)
	defer rt.Close()
	defer sess.Close()

	input := make([]float32, 1*3*640*640)
	run := sess.NewRun()
	defer run.Close()

	if err := onnx.BindSlice(run, "images", input, []int64{1, 3, 640, 640}); err != nil {
		t.Fatalf("BindSlice: %v", err)
	}

	// Warm up
	if err := run.Exec(); err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if _, err := run.GetOutput(0); err != nil {
		t.Fatalf("GetOutput: %v", err)
	}

	// Measure allocations over N iterations
	const N = 10
	allocs := testing.AllocsPerRun(N, func() {
		if err := run.Exec(); err != nil {
			t.Fatalf("Exec: %v", err)
		}
		out, err := run.GetOutput(0)
		if err != nil {
			t.Fatalf("GetOutput: %v", err)
		}
		_ = unsafe.Slice((*float32)(out.Ptr), out.Len)
	})

	fmt.Printf("Allocations per Exec+GetOutput: %.1f\n", allocs)
	if allocs > 0 {
		t.Fatalf("expected 0 allocations in hot loop, got %.1f", allocs)
	}
}
