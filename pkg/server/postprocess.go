package server

import (
	"math"
	"slices"

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

// yoloDet is a box decoded from YOLO output, in model-input pixels; anchor
// is its column in the output.
type yoloDet struct {
	x1, y1, x2, y2 float32
	classIdx       int
	conf           float32
	anchor         int
}

// yoloScratch is what decoding a YOLO output works in, kept to decode the
// next one without allocating.
type yoloScratch struct {
	hit  []yoloDet
	best []int32 // per anchor: its index in hit + 1; 0: none
	keep []bool
}

// decode decodes YOLO-style [1, 4+C(+M), N] output, the class scores being
// the nClasses channels after the box, with NMS. The result is valid until
// the next decode.
func (s *yoloScratch) decode(data []float32, shape []int64, nClasses int) []yoloDet {
	nDets := int(shape[2])
	const confThreshold = 0.25
	const iouThreshold = 0.45

	if cap(s.best) < nDets {
		s.best = make([]int32, nDets)
	}
	best := s.best[:nDets]
	// Scan each class column for scores above the threshold (nearly all are
	// below), keeping every anchor's best. A column is contiguous, which is
	// several times faster than walking all classes of each anchor.
	hit := s.hit[:0]
	for c := 0; c < nClasses; c++ {
		col := data[(4+c)*nDets : (5+c)*nDets]
		for d, v := range col {
			if v < confThreshold {
				continue
			}
			if i := best[d]; i > 0 {
				if v > hit[i-1].conf {
					hit[i-1].conf, hit[i-1].classIdx = v, c
				}
				continue
			}
			hit = append(hit, yoloDet{classIdx: c, conf: v, anchor: d})
			best[d] = int32(len(hit)) //nolint:gosec // at most nDets
		}
	}
	for _, h := range hit {
		best[h.anchor] = 0 // clean for the next decode
	}
	s.hit = hit
	// Anchor order, as a plain scan produces, for a stable sort below.
	slices.SortFunc(hit, func(a, b yoloDet) int { return a.anchor - b.anchor })
	dets := hit
	for i := range dets {
		d := dets[i].anchor
		cx, cy, w, h := data[d], data[nDets+d], data[2*nDets+d], data[3*nDets+d]
		dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
	}

	// Sort descending by confidence
	for i := 1; i < len(dets); i++ {
		for j := i; j > 0 && dets[j].conf > dets[j-1].conf; j-- {
			dets[j], dets[j-1] = dets[j-1], dets[j]
		}
	}

	// NMS per class
	if cap(s.keep) < len(dets) {
		s.keep = make([]bool, len(dets))
	}
	keep := s.keep[:len(dets)]
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
	kept := dets[:0]
	for i, d := range dets {
		if keep[i] {
			kept = append(kept, d)
		}
	}
	return kept
}

// yoloDetection maps d back to the original image.
func yoloDetection(d yoloDet, lbls labels.Labels, imgTensor *goimage.Tensor) *gomaxv1.Detection {
	label := ""
	if lbls != nil {
		label = lbls.Get(d.classIdx)
	}
	ox1 := (d.x1 - float32(imgTensor.PadX)) / float32(imgTensor.Scale)
	oy1 := (d.y1 - float32(imgTensor.PadY)) / float32(imgTensor.Scale)
	ox2 := (d.x2 - float32(imgTensor.PadX)) / float32(imgTensor.Scale)
	oy2 := (d.y2 - float32(imgTensor.PadY)) / float32(imgTensor.Scale)
	return &gomaxv1.Detection{
		Label:      label,
		ClassId:    int32(d.classIdx),
		Confidence: d.conf,
		X1:         ox1, Y1: oy1, X2: ox2, Y2: oy2,
	}
}

// DetectOutput decodes YOLO-style [1, 4+C, N] output with NMS and coordinate mapping.
func DetectOutput(data []float32, shape []int64, lbls labels.Labels, imgTensor *goimage.Tensor) *gomaxv1.Detections {
	return detectOutput(new(yoloScratch), data, shape, lbls, imgTensor)
}

// detectOutput is DetectOutput, decoding in s.
func detectOutput(s *yoloScratch, data []float32, shape []int64, lbls labels.Labels, imgTensor *goimage.Tensor) *gomaxv1.Detections {
	nClasses := int(shape[1]) - 4
	// A segmentation model (YOLO -seg) has its mask coefficients after the
	// class scores: with fewer labels than channels, only the labels' are
	// classes.
	if n := len(lbls); n > 0 && n < nClasses {
		nClasses = n
	}
	items := make([]*gomaxv1.Detection, 0)
	for _, d := range s.decode(data, shape, nClasses) {
		items = append(items, yoloDetection(d, lbls, imgTensor))
	}
	return &gomaxv1.Detections{Items: items}
}

// SegmentOutput decodes YOLO -seg output: data [1, 4+C+M, N] as for
// DetectOutput, each detection's last M channels its coefficients for the
// M mask prototypes protos [1, M, H, W] (H×W a quarter of the model input).
// A detection's mask is sigmoid(coefficients · prototypes) > 0.5, upsampled
// to the original image and cut to the detection's box.
func SegmentOutput(data []float32, shape []int64, protos []float32, protoShape []int64, lbls labels.Labels, imgTensor *goimage.Tensor) *gomaxv1.Detections {
	return segmentOutput(new(yoloScratch), data, shape, protos, protoShape, lbls, imgTensor)
}

// segmentOutput is SegmentOutput, decoding in s.
func segmentOutput(s *yoloScratch, data []float32, shape []int64, protos []float32, protoShape []int64, lbls labels.Labels, imgTensor *goimage.Tensor) *gomaxv1.Detections {
	nm := int(protoShape[1])
	ph, pw := int(protoShape[2]), int(protoShape[3])
	nClasses := int(shape[1]) - 4 - nm
	nDets := int(shape[2])
	// Model-input pixels per prototype pixel.
	sx := float32(imgTensor.Shape[3]) / float32(pw)
	sy := float32(imgTensor.Shape[2]) / float32(ph)

	items := make([]*gomaxv1.Detection, 0)
	coef := make([]float32, nm)
	for _, d := range s.decode(data, shape, nClasses) {
		det := yoloDetection(d, lbls, imgTensor)
		for k := range coef {
			coef[k] = data[(4+nClasses+k)*nDets+d.anchor]
		}
		det.Mask = yoloMask(coef, protos, pw, ph, sx, sy, det, imgTensor)
		items = append(items, det)
	}
	return &gomaxv1.Detections{Items: items}
}

// yoloMask is the mask of det, in original-image pixels over its box.
func yoloMask(coef, protos []float32, pw, ph int, sx, sy float32, det *gomaxv1.Detection, imgTensor *goimage.Tensor) *gomaxv1.Mask {
	// The box in original-image pixels, clamped to the image.
	x0 := clampInt(int(math.Floor(float64(det.X1))), 0, imgTensor.SrcW)
	y0 := clampInt(int(math.Floor(float64(det.Y1))), 0, imgTensor.SrcH)
	x1 := clampInt(int(math.Ceil(float64(det.X2))), 0, imgTensor.SrcW)
	y1 := clampInt(int(math.Ceil(float64(det.Y2))), 0, imgTensor.SrcH)
	if x1 <= x0 || y1 <= y0 {
		return nil
	}
	w, h := x1-x0, y1-y0
	scale := float32(imgTensor.Scale)
	padX, padY := float32(imgTensor.PadX), float32(imgTensor.PadY)
	// protoXY is the prototype coordinate of an original-image pixel's
	// center, prototype pixel centers being at integers.
	protoX := func(x int) float32 { return ((float32(x)+0.5)*scale+padX)/sx - 0.5 }
	protoY := func(y int) float32 { return ((float32(y)+0.5)*scale+padY)/sy - 0.5 }

	// Mask logits over the prototype pixels under the box (and one around,
	// to interpolate at its edge).
	gx0 := clampInt(int(math.Floor(float64(protoX(x0)))), 0, pw-1)
	gy0 := clampInt(int(math.Floor(float64(protoY(y0)))), 0, ph-1)
	gx1 := clampInt(int(math.Floor(float64(protoX(x1-1))))+1, 0, pw-1)
	gy1 := clampInt(int(math.Floor(float64(protoY(y1-1))))+1, 0, ph-1)
	gw, gh := gx1-gx0+1, gy1-gy0+1
	logits := make([]float32, gw*gh)
	plane := pw * ph
	for k, c := range coef {
		p := protos[k*plane:]
		for gy := 0; gy < gh; gy++ {
			row := p[(gy0+gy)*pw+gx0:]
			out := logits[gy*gw : (gy+1)*gw]
			for gx := range out {
				out[gx] += c * row[gx]
			}
		}
	}
	at := func(gx, gy int) float32 {
		return logits[clampInt(gy-gy0, 0, gh-1)*gw+clampInt(gx-gx0, 0, gw-1)]
	}

	// Bilinear upsampling of the logits; sigmoid > 0.5 is logit > 0.
	bits := make([]byte, (w*h+7)/8)
	i := 0
	for y := y0; y < y1; y++ {
		fy := protoY(y)
		iy := int(math.Floor(float64(fy)))
		ty := fy - float32(iy)
		for x := x0; x < x1; x++ {
			fx := protoX(x)
			ix := int(math.Floor(float64(fx)))
			tx := fx - float32(ix)
			top := at(ix, iy)*(1-tx) + at(ix+1, iy)*tx
			bot := at(ix, iy+1)*(1-tx) + at(ix+1, iy+1)*tx
			if top*(1-ty)+bot*ty > 0 {
				bits[i>>3] |= 0x80 >> (i & 7)
			}
			i++
		}
	}
	return &gomaxv1.Mask{X: uint32(x0), Y: uint32(y0), Width: uint32(w), Height: uint32(h), Bits: bits}
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// NormalizeCHW applies per-channel (x - mean) / std in-place on CHW [0,1] data.
func NormalizeCHW(data []float32, width, height int, mean, std []float32) {
	chSize := width * height
	for c := 0; c < 3 && c < len(mean) && c < len(std); c++ {
		off := c * chSize
		m, s := mean[c], std[c]
		for i := 0; i < chSize; i++ {
			data[off+i] = (data[off+i] - m) / s
		}
	}
}

// DetrDetectOutput decodes DETR-style [1, Q, C] logits + [1, Q, 4] normalized [cx,cy,w,h] boxes.
func DetrDetectOutput(logits []float32, logitsShape []int64, boxes []float32, lbls labels.Labels, imgTensor *goimage.Tensor) *gomaxv1.Detections {
	numQueries := int(logitsShape[1])
	numClasses := int(logitsShape[2])
	const confThreshold = 0.35
	const iouThreshold = 0.7

	type det struct {
		x1, y1, x2, y2 float32
		classIdx       int
		conf           float32
	}

	var imgW, imgH float32
	if imgTensor != nil {
		imgW = float32(imgTensor.Shape[3])
		imgH = float32(imgTensor.Shape[2])
	} else {
		imgW, imgH = 1, 1
	}

	var dets []det
	for q := 0; q < numQueries; q++ {
		base := q * numClasses
		bestClass := -1
		var bestLogit float32
		// Start from class 1 to skip the background/no-object class at index 0.
		for c := 1; c < numClasses; c++ {
			if bestClass < 0 || logits[base+c] > bestLogit {
				bestLogit = logits[base+c]
				bestClass = c
			}
		}
		if bestClass < 0 {
			continue
		}
		conf := sigmoid32(bestLogit)
		if conf < confThreshold {
			continue
		}

		bbase := q * 4
		cx := boxes[bbase+0]
		cy := boxes[bbase+1]
		w := boxes[bbase+2]
		h := boxes[bbase+3]

		x1 := (cx - w/2) * imgW
		y1 := (cy - h/2) * imgH
		x2 := (cx + w/2) * imgW
		y2 := (cy + h/2) * imgH

		dets = append(dets, det{x1, y1, x2, y2, bestClass, conf})
	}

	// Sort descending by confidence.
	for i := 1; i < len(dets); i++ {
		for j := i; j > 0 && dets[j].conf > dets[j-1].conf; j-- {
			dets[j], dets[j-1] = dets[j-1], dets[j]
		}
	}

	// NMS per class.
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
		ox1, oy1, ox2, oy2 := d.x1, d.y1, d.x2, d.y2
		if imgTensor != nil {
			ox1 = (d.x1 - float32(imgTensor.PadX)) / float32(imgTensor.Scale)
			oy1 = (d.y1 - float32(imgTensor.PadY)) / float32(imgTensor.Scale)
			ox2 = (d.x2 - float32(imgTensor.PadX)) / float32(imgTensor.Scale)
			oy2 = (d.y2 - float32(imgTensor.PadY)) / float32(imgTensor.Scale)
		}
		items = append(items, &gomaxv1.Detection{
			Label:      label,
			ClassId:    int32(d.classIdx),
			Confidence: d.conf,
			X1:         ox1, Y1: oy1, X2: ox2, Y2: oy2,
		})
	}
	return &gomaxv1.Detections{Items: items}
}

// OvDetrDetectOutput decodes open-vocabulary DETR output: per-class queries with
// independent scores. Expects scores shape [C, Q, 1] (one sigmoid logit per
// (class, query) pair) and boxes shape [C, Q, 4] (normalized [cx,cy,w,h]).
// Unlike DetrDetectOutput, the class index is the outer dim — not an argmax across
// a class axis — so each class predicts its own boxes independently.
func OvDetrDetectOutput(scores []float32, scoresShape []int64, boxes []float32, lbls labels.Labels, imgTensor *goimage.Tensor) *gomaxv1.Detections {
	numClasses := int(scoresShape[0])
	numQueries := int(scoresShape[1])
	const confThreshold = 0.35
	const iouThreshold = 0.7

	var imgW, imgH float32
	if imgTensor != nil {
		imgW = float32(imgTensor.Shape[3])
		imgH = float32(imgTensor.Shape[2])
	} else {
		imgW, imgH = 1, 1
	}

	type det struct {
		x1, y1, x2, y2 float32
		classIdx       int
		conf           float32
	}

	var dets []det
	for c := 0; c < numClasses; c++ {
		for q := 0; q < numQueries; q++ {
			conf := sigmoid32(scores[c*numQueries+q])
			if conf < confThreshold {
				continue
			}
			bbase := (c*numQueries + q) * 4
			cx := boxes[bbase+0]
			cy := boxes[bbase+1]
			w := boxes[bbase+2]
			h := boxes[bbase+3]
			dets = append(dets, det{
				x1:       (cx - w/2) * imgW,
				y1:       (cy - h/2) * imgH,
				x2:       (cx + w/2) * imgW,
				y2:       (cy + h/2) * imgH,
				classIdx: c,
				conf:     conf,
			})
		}
	}

	// Sort descending by confidence.
	for i := 1; i < len(dets); i++ {
		for j := i; j > 0 && dets[j].conf > dets[j-1].conf; j-- {
			dets[j], dets[j-1] = dets[j-1], dets[j]
		}
	}

	// NMS per class.
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
		ox1, oy1, ox2, oy2 := d.x1, d.y1, d.x2, d.y2
		if imgTensor != nil {
			ox1 = (d.x1 - float32(imgTensor.PadX)) / float32(imgTensor.Scale)
			oy1 = (d.y1 - float32(imgTensor.PadY)) / float32(imgTensor.Scale)
			ox2 = (d.x2 - float32(imgTensor.PadX)) / float32(imgTensor.Scale)
			oy2 = (d.y2 - float32(imgTensor.PadY)) / float32(imgTensor.Scale)
		}
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
