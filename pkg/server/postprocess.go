package server

import (
	"math"

	goimage "github.com/aep/moxgo/pkg/image"
	"github.com/aep/moxgo/pkg/labels"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
)

func sigmoid32(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// ClassifyOutput returns top-K classifications. If sigmoid > 0, applies sigmoid(sensitivity * x) to logits.
func ClassifyOutput(data []float32, lbls labels.Labels, topK int, sigmoid float64) *gomaxv1.Classifications {
	if topK > len(data) {
		topK = len(data)
	}
	used := make([]bool, len(data))
	items := make([]*gomaxv1.Classification, 0, topK)
	for range topK {
		bestIdx := -1
		var bestVal float32
		for j, v := range data {
			if used[j] {
				continue
			}
			if bestIdx == -1 || v > bestVal {
				bestIdx = j
				bestVal = v
			}
		}
		if bestIdx < 0 {
			break
		}
		used[bestIdx] = true
		label := ""
		if lbls != nil {
			label = lbls.Get(bestIdx)
		}
		score := bestVal
		if sigmoid > 0 {
			score = sigmoid32(float32(sigmoid) * score)
		}
		items = append(items, &gomaxv1.Classification{
			Label:   label,
			ClassId: int32(bestIdx),
			Score:   score,
		})
	}
	return &gomaxv1.Classifications{Items: items}
}

// DetectOutput decodes YOLO [1, 4+C, N] output with NMS and coordinate mapping.
func DetectOutput(data []float32, shape []int64, lbls labels.Labels, imgTensor *goimage.Tensor) *gomaxv1.Detections {
	nClasses := int(shape[1]) - 4
	nDets := int(shape[2])
	const confThreshold = 0.25
	const iouThreshold = 0.45

	type det struct {
		x1, y1, x2, y2 float32
		classIdx       int
		conf           float32
	}

	var dets []det
	for d := 0; d < nDets; d++ {
		bestClass := 0
		bestConf := data[4*nDets+d]
		for c := 1; c < nClasses; c++ {
			score := data[(4+c)*nDets+d]
			if score > bestConf {
				bestConf = score
				bestClass = c
			}
		}
		if bestConf < confThreshold {
			continue
		}
		cx := data[0*nDets+d]
		cy := data[1*nDets+d]
		w := data[2*nDets+d]
		h := data[3*nDets+d]
		dets = append(dets, det{cx - w/2, cy - h/2, cx + w/2, cy + h/2, bestClass, bestConf})
	}

	// Sort descending by confidence
	for i := 1; i < len(dets); i++ {
		for j := i; j > 0 && dets[j].conf > dets[j-1].conf; j-- {
			dets[j], dets[j-1] = dets[j-1], dets[j]
		}
	}

	// NMS per class
	keep := make([]bool, len(dets))
	for i := range keep {
		keep[i] = true
	}
	for i := 0; i < len(dets); i++ {
		if !keep[i] {
			continue
		}
		for j := i + 1; j < len(dets); j++ {
			if !keep[j] || dets[j].classIdx != dets[i].classIdx {
				continue
			}
			if bboxIoU(dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2,
				dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2) > iouThreshold {
				keep[j] = false
			}
		}
	}

	items := make([]*gomaxv1.Detection, 0)
	for i, d := range dets {
		if !keep[i] {
			continue
		}
		label := ""
		if lbls != nil {
			label = lbls.Get(d.classIdx)
		}
		ox1 := (d.x1 - float32(imgTensor.PadX)) / float32(imgTensor.Scale)
		oy1 := (d.y1 - float32(imgTensor.PadY)) / float32(imgTensor.Scale)
		ox2 := (d.x2 - float32(imgTensor.PadX)) / float32(imgTensor.Scale)
		oy2 := (d.y2 - float32(imgTensor.PadY)) / float32(imgTensor.Scale)
		items = append(items, &gomaxv1.Detection{
			Label:      label,
			ClassId:    int32(d.classIdx),
			Confidence: d.conf,
			X1:         ox1, Y1: oy1, X2: ox2, Y2: oy2,
		})
	}
	return &gomaxv1.Detections{Items: items}
}

// EmbeddingOutput wraps raw output as an Embedding proto.
func EmbeddingOutput(data []float32, shape []int64) *gomaxv1.Embedding {
	values := make([]float32, len(data))
	copy(values, data)
	s := make([]int64, len(shape))
	copy(s, shape)
	return &gomaxv1.Embedding{Values: values, Shape: s}
}

// RawOutput wraps raw output as a RawTensor proto.
func RawOutput(data []float32, shape []int64) *gomaxv1.RawTensor {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		buf[i*4] = byte(bits)
		buf[i*4+1] = byte(bits >> 8)
		buf[i*4+2] = byte(bits >> 16)
		buf[i*4+3] = byte(bits >> 24)
	}
	s := make([]int64, len(shape))
	copy(s, shape)
	return &gomaxv1.RawTensor{Data: buf, Shape: s}
}

// IsClassificationShape returns true for [1,N], [-1,N], or [N].
func IsClassificationShape(shape []int64) bool {
	switch len(shape) {
	case 1:
		return shape[0] > 1 || shape[0] == -1
	case 2:
		return (shape[0] == 1 || shape[0] == -1) && (shape[1] > 1 || shape[1] == -1)
	default:
		return false
	}
}

// IsYOLODetectionShape returns true for [1, 4+C, N].
func IsYOLODetectionShape(shape []int64) bool {
	if len(shape) != 3 {
		return false
	}
	return shape[0] >= 1 && shape[1] > 4 && shape[2] > shape[1]
}

// ClassCountFromShape extracts class count from a classification shape.
func ClassCountFromShape(shape []int64) int {
	switch len(shape) {
	case 1:
		if shape[0] > 0 {
			return int(shape[0])
		}
	case 2:
		if shape[1] > 0 {
			return int(shape[1])
		}
	}
	return 0
}

func bboxIoU(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 float32) float32 {
	ix1 := max32(ax1, bx1)
	iy1 := max32(ay1, by1)
	ix2 := min32(ax2, bx2)
	iy2 := min32(ay2, by2)
	if ix2 <= ix1 || iy2 <= iy1 {
		return 0
	}
	inter := (ix2 - ix1) * (iy2 - iy1)
	areaA := (ax2 - ax1) * (ay2 - ay1)
	areaB := (bx2 - bx1) * (by2 - by1)
	return inter / (areaA + areaB - inter)
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
