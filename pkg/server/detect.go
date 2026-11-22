package server

import (
	"context"
	"errors"
	"fmt"
	"unsafe"

	goimage "github.com/aep/moxgo/pkg/image"
	"github.com/aep/moxgo/pkg/onnx"
)

// Detection is an object a detection model found in a picture: its label
// and class, the model's confidence, and its box, corners X1,Y1 to X2,Y2,
// in pixels of the picture.
type Detection struct {
	Label      string
	ClassID    int32
	Confidence float32
	X1, Y1     float32
	X2, Y2     float32
}

// ErrNotDetector: the model is not one Detect runs (see Detect). Use
// PredictContext for it.
var ErrNotDetector = errors.New("not a detection model Detect runs")

// Detect runs a detection model on a decoded picture and appends what it
// found to dst, which it returns. It is the allocation-free way to run a
// detector on video frames: no protobuf in or out, src read in place, and
// what it works in kept with the model's run slots, so a caller that passes
// the slice it got the last time allocates nothing once the model is warm.
// Labels are the model's own strings, not copies.
//
// It runs models with a single image input and a YOLO detection output (the
// "yolo" decoder): ErrNotDetector for others, which PredictContext runs.
// Traced under ctx's span like PredictContext.
func (inst *Instance) Detect(ctx context.Context, src *goimage.Planes, dst []Detection) (_ []Detection, err error) {
	ctx, span := inst.startPredict(ctx)
	if span.IsRecording() {
		defer func() {
			if err != nil {
				span.RecordError(err)
			}
			span.End()
		}()
	}
	if !inst.begin() {
		return dst, fmt.Errorf("model %s is stopped", inst.Name)
	}
	defer inst.use.RUnlock()

	out, oc := inst.detectOutput()
	if out < 0 {
		return dst, ErrNotDetector
	}
	inp := inst.Config.InputList[0]
	if src == nil || src.Width <= 0 || src.Height <= 0 {
		return dst, inputErrorf("image %q: no picture", inp.Name)
	}

	_, wait := startSpan(ctx, "moxgo.wait_slot")
	slot := inst.acquireSlot()
	wait.End()
	defer inst.releaseSlot(slot)

	_, pre := startSpan(ctx, "moxgo.preprocess")
	pp := slot.preprocessor(0, inp)
	pp.Invalidate() // the slot may have held another picture
	lb := pp.Process(slot.bufs[0], src)
	pre.End()

	if !slot.warmed {
		_, warm := startSpan(ctx, "moxgo.warmup")
		err := inst.warmSlot(slot)
		warm.End()
		if err != nil {
			return dst, err
		}
	}

	_, infer := startSpan(ctx, "moxgo.inference")
	err = slot.run.Exec()
	infer.End()
	if err != nil {
		return dst, fmt.Errorf("exec: %w", err)
	}

	_, post := startSpan(ctx, "moxgo.postprocess")
	defer post.End()
	result, err := slot.run.GetOutput(out)
	if err != nil {
		return dst, fmt.Errorf("output: %w", err)
	}
	if result.Dtype != onnx.ElemTypeFloat32 || len(result.Shape) != 3 || result.Len == 0 {
		return dst, fmt.Errorf("output: not a YOLO detection tensor (%v)", result.Shape)
	}
	data := unsafe.Slice((*float32)(result.Ptr), result.Len)
	nClasses := int(result.Shape[1]) - 4
	if n := len(oc.ResolvedLabels); n > 0 && n < nClasses {
		nClasses = n // a -seg model's mask coefficients follow its classes
	}
	scale := float32(lb.Scale)
	padX, padY := float32(lb.PadX), float32(lb.PadY)
	for _, d := range slot.yolo.decode(data, result.Shape, nClasses) {
		label := ""
		if oc.ResolvedLabels != nil {
			label = oc.ResolvedLabels.Get(d.classIdx)
		}
		dst = append(dst, Detection{
			Label: label, ClassID: int32(d.classIdx), Confidence: d.conf, //nolint:gosec // a class index
			X1: (d.x1 - padX) / scale, Y1: (d.y1 - padY) / scale,
			X2: (d.x2 - padX) / scale, Y2: (d.y2 - padY) / scale,
		})
	}
	return dst, nil
}

// detectOutput is the index of the session output Detect decodes, and its
// configuration; -1 for a model Detect does not run.
func (inst *Instance) detectOutput() (int, *OutputConfig) {
	if inst.sam3 != nil || len(inst.Config.InputList) != 1 || inst.Config.InputList[0].Type != "image" {
		return -1, nil
	}
	for i, out := range inst.Session.Outputs {
		oc := inst.Config.Outputs[out.Name]
		// as buildOutputResult: a decoder other than detr and ov is YOLO
		if oc != nil && oc.Type == "detection" && oc.Decoder != "detr" && oc.Decoder != "ov" {
			return i, oc
		}
	}
	return -1, nil
}
