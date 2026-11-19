package onnx

import (
	"os"
	"runtime"
	"sync"
	"testing"
)

//go:noinline
func atDepth(d int, fn func()) byte {
	var pad [128]byte
	if d == 0 {
		fn()
	} else {
		pad[d%128] = atDepth(d-1, fn)
	}
	return pad[d%128]
}

//go:noinline
func smallFrames(d int, fn func()) byte {
	var pad [8]byte
	if d == 0 {
		fn()
	} else {
		pad[d%8] = smallFrames(d-1, fn)
	}
	return pad[d%8]
}

// ONNX Runtime writes results through pointers it is given. If such a pointer
// is to a goroutine stack that moves on the way into the call, the result is
// lost: Bind then yields a NULL input, which made a model fail its first run
// now and then and silently fall back to the CPU. Fresh goroutines at varied
// depths make their stacks grow at every point of the call path.
func TestResultsSurviveStackGrowth(t *testing.T) {
	const model = "../../models/yolo11n.onnx"
	if _, err := os.Stat(model); err != nil {
		t.Skipf("model not found: %s", model)
	}
	rt, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	sess, err := rt.OpenSession(model)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	input := make([]float32, 3*640*640)
	shape := []int64{1, 3, 640, 640}
	var mu sync.Mutex
	lost := 0
	// The collector shrinks stacks that are mostly unused, which moves them too.
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()
	for round := 0; round < 60; round++ {
		var wg sync.WaitGroup
		for depth := 0; depth < 200; depth++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				atDepth(depth%40, func() {
					smallFrames(depth, func() {
						r := sess.NewRun()
						defer r.Close()
						if err := BindSlice(r, "images", input, shape); err != nil {
							t.Error(err)
							return
						}
						if r.inputValues[0] == 0 {
							mu.Lock()
							lost++
							mu.Unlock()
						}
					})
				})
			}()
		}
		wg.Wait()
	}
	if lost > 0 {
		t.Fatalf("%d of 12000 binds lost their tensor", lost)
	}
}
