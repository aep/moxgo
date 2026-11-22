package local

import (
	"context"
	"image"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"testing"

	goimage "github.com/aep/moxgo/pkg/image"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
)

// BenchmarkPredictYCbCr runs a detector on a 640×360 4:2:0 picture, as an
// embedding program hands moxgo a decoded video frame: the time and the
// allocations of one prediction. Needs ONNX Runtime and the model pulled
// (moxgo pull yolo11n); skipped without.
func BenchmarkPredictYCbCr(b *testing.B) {
	srv, rt, err := New(Options{})
	if err != nil {
		b.Skipf("no ONNX Runtime: %v", err)
	}
	defer rt.Close()
	defer srv.Manager().Shutdown()
	const model = "yolo11n"
	if _, err := srv.Manager().GetOrRun(model); err != nil {
		b.Skipf("model %s: %v", model, err)
	}
	inst, err := srv.Manager().GetOrRun(model)
	if err != nil {
		b.Fatal(err)
	}
	w, h := 640, 360
	in := []*gomaxv1.Input{{Name: "image", Data: &gomaxv1.Input_Ycbcr{Ycbcr: &gomaxv1.YCbCrImage{
		Width: uint32(w), Height: uint32(h),
		Y: make([]byte, w*h), Cb: make([]byte, w*h/4), Cr: make([]byte, w*h/4),
	}}}}
	ctx := context.Background()
	if _, err := inst.PredictContext(ctx, in); err != nil { // warm up
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := inst.PredictContext(ctx, in); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDetect is BenchmarkPredictYCbCr through Detect, which takes the
// picture in place and appends to the caller's slice: no allocations.
func BenchmarkDetect(b *testing.B) {
	srv, rt, err := New(Options{})
	if err != nil {
		b.Skipf("no ONNX Runtime: %v", err)
	}
	defer rt.Close()
	defer srv.Manager().Shutdown()
	inst, err := srv.Manager().GetOrRun("yolo11n")
	if err != nil {
		b.Skipf("model yolo11n: %v", err)
	}
	w, h := 640, 360
	src := &goimage.Planes{Width: w, Height: h, Y: make([]byte, w*h), YStride: w,
		U: make([]byte, w*h/4), V: make([]byte, w*h/4), UVStride: w / 2}
	ctx := context.Background()
	dets, err := inst.Detect(ctx, src, nil) // warm up
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if dets, err = inst.Detect(ctx, src, dets[:0]); err != nil {
			b.Fatal(err)
		}
	}
}

// TestDetectMatchesPredict: Detect finds what PredictContext finds in the
// same picture, box for box.
func TestDetectMatchesPredict(t *testing.T) {
	srv, rt, err := New(Options{})
	if err != nil {
		t.Skipf("no ONNX Runtime: %v", err)
	}
	defer rt.Close()
	defer srv.Manager().Shutdown()
	inst, err := srv.Manager().GetOrRun("yolo11n")
	if err != nil {
		t.Skipf("model yolo11n: %v", err)
	}
	for _, name := range []string{"people.jpg", "traffic.jpg", "street.jpg"} {
		f, err := os.Open(filepath.Join("..", "..", "testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		pic, _, err := image.Decode(f)
		_ = f.Close()
		if err != nil {
			t.Fatal(err)
		}
		yc, ok := pic.(*image.YCbCr)
		if !ok || yc.SubsampleRatio != image.YCbCrSubsampleRatio420 {
			t.Skipf("%s is not a 4:2:0 JPEG", name)
		}
		planes, _ := goimage.PlanesFromYCbCr(yc)
		w, h := yc.Rect.Dx(), yc.Rect.Dy()
		in := &gomaxv1.YCbCrImage{Width: uint32(w), Height: uint32(h)}
		for y := range h {
			in.Y = append(in.Y, yc.Y[y*yc.YStride:y*yc.YStride+w]...)
		}
		for y := range (h + 1) / 2 {
			in.Cb = append(in.Cb, yc.Cb[y*yc.CStride:y*yc.CStride+(w+1)/2]...)
			in.Cr = append(in.Cr, yc.Cr[y*yc.CStride:y*yc.CStride+(w+1)/2]...)
		}
		resp, err := inst.PredictContext(context.Background(), []*gomaxv1.Input{{Name: "image", Data: &gomaxv1.Input_Ycbcr{Ycbcr: in}}})
		if err != nil {
			t.Fatal(err)
		}
		var want []*gomaxv1.Detection
		for _, o := range resp.GetOutputs() {
			want = append(want, o.GetDetections().GetItems()...)
		}
		got, err := inst.Detect(context.Background(), &planes, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want) {
			t.Fatalf("%s: Detect found %d, PredictContext %d", name, len(got), len(want))
		}
		for i, w := range want {
			g := got[i]
			if g.Label != w.GetLabel() || g.ClassID != w.GetClassId() || g.Confidence != w.GetConfidence() ||
				g.X1 != w.GetX1() || g.Y1 != w.GetY1() || g.X2 != w.GetX2() || g.Y2 != w.GetY2() {
				t.Fatalf("%s: detection %d: %+v, want %v", name, i, g, w)
			}
		}
		t.Logf("%s: %d detections, the same", name, len(got))
	}
}
