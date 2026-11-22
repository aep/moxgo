package image

import (
	"image"
	"runtime"
	"sync"
)

// Planes is a YCbCr 4:2:0 picture as a video decoder hands it out, in place:
// nothing is copied or converted to read it.
type Planes struct {
	Width, Height int
	Y             []byte
	YStride       int
	// U and V are the chroma planes. With V nil, U holds interleaved UV (NV12).
	U, V     []byte
	UVStride int
	// Limited says luma spans 16-235 and chroma 16-240, as in most video. The
	// zero value is full range, as in JPEG.
	Limited bool
	// BT709 selects that matrix. The zero value is BT.601, as in JPEG.
	BT709 bool
}

// PlanesFromYCbCr views a 4:2:0 image. ok is false for other subsamplings.
func PlanesFromYCbCr(img *image.YCbCr) (p Planes, ok bool) {
	if img.SubsampleRatio != image.YCbCrSubsampleRatio420 {
		return p, false
	}
	b := img.Rect
	return Planes{
		Width: b.Dx(), Height: b.Dy(),
		Y:       img.Y[img.YOffset(b.Min.X, b.Min.Y):],
		YStride: img.YStride,
		U:       img.Cb[img.COffset(b.Min.X, b.Min.Y):],
		V:       img.Cr[img.COffset(b.Min.X, b.Min.Y):],

		UVStride: img.CStride,
	}, true
}

// Letterbox says where a picture landed in a model input.
type Letterbox struct {
	Scale              float64 // model-input pixels per source pixel
	PadX, PadY         int
	ContentW, ContentH int // the picture's size in the model input
	SrcW, SrcH         int
}

// ComputeLetterbox fits srcW×srcH into width×height keeping the aspect ratio.
func ComputeLetterbox(srcW, srcH, width, height int) Letterbox {
	scale := float64(width) / float64(srcW)
	if s := float64(height) / float64(srcH); s < scale {
		scale = s
	}
	newW, newH := int(float64(srcW)*scale), int(float64(srcH)*scale)
	return Letterbox{
		Scale: scale, PadX: (width - newW) / 2, PadY: (height - newH) / 2,
		ContentW: newW, ContentH: newH, SrcW: srcW, SrcH: srcH,
	}
}

// Tensor returns the transform as a Tensor without data, in the form model
// output decoders take it. Coordinates they produce are source pixels.
func (l Letterbox) Tensor(width, height int) *Tensor {
	return &Tensor{
		Shape: []int64{1, 3, int64(height), int64(width)},
		Scale: l.Scale, PadX: l.PadX, PadY: l.PadY, SrcW: l.SrcW, SrcH: l.SrcH,
	}
}

// ContentTensor is Tensor with the picture's size in the model input as the
// source size: decoders then produce coordinates (and masks) in those pixels,
// which costs less than source pixels when the source is large.
func (l Letterbox) ContentTensor(width, height int) *Tensor {
	return &Tensor{
		Shape: []int64{1, 3, int64(height), int64(width)},
		Scale: 1, PadX: l.PadX, PadY: l.PadY, SrcW: l.ContentW, SrcH: l.ContentH,
	}
}

// padValue is the letterbox border, before normalization.
const padValue = 114.0 / 255.0

const clampOffset = 384 // conversion results stay within -384..639

// Preprocessor turns pictures into CHW float32 model inputs without going
// through RGB images: it scales the planes it needs straight into the tensor.
// It is not safe for concurrent use.
type Preprocessor struct {
	w, h    int
	workers int

	// channel values for each 8-bit result, clamped and normalized
	r, g, b [1024]float32
	pad     [3]float32

	// YCbCr to RGB tables for the current color description, Q16
	colorSet bool
	limited  bool
	bt709    bool
	yt       [256]int32
	rv       [256]int32
	gu       [256]int32
	gv       [256]int32
	bu       [256]int32

	// geometry of the last picture
	lb       Letterbox
	sy, su   sampler
	padDone  *float32
	rowPools [][]byte

	// current job
	src *Planes
	dst []float32
	row func(worker, lo, hi int)
	wg  sync.WaitGroup
}

// NewPreprocessor makes a Preprocessor for width×height inputs. mean and std,
// if set, normalize each channel as (v-mean)/std on values in [0,1].
func NewPreprocessor(width, height int, mean, std []float32) *Preprocessor {
	p := &Preprocessor{w: width, h: height, workers: preprocessWorkers()}
	m, s := [3]float32{}, [3]float32{1, 1, 1}
	if len(mean) >= 3 && len(std) >= 3 {
		copy(m[:], mean)
		copy(s[:], std)
	}
	for c, t := range []*[1024]float32{&p.r, &p.g, &p.b} {
		for i := range t {
			v := i - clampOffset
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			t[i] = (float32(v)/255 - m[c]) / s[c]
		}
		p.pad[c] = (padValue - m[c]) / s[c]
	}
	p.rowPools = make([][]byte, p.workers)
	for i := range p.rowPools {
		p.rowPools[i] = make([]byte, 3*width)
	}
	p.row = p.convertRows
	return p
}

// preprocessWorkers is how many goroutines share a picture: enough for a few
// milliseconds, not so many that they fight the video goroutines.
func preprocessWorkers() int {
	return max(1, min(4, runtime.GOMAXPROCS(0)/2))
}

func (p *Preprocessor) setColor(limited, bt709 bool) {
	if p.colorSet && p.limited == limited && p.bt709 == bt709 {
		return
	}
	p.colorSet, p.limited, p.bt709 = true, limited, bt709
	kr, kb := 0.299, 0.114
	if bt709 {
		kr, kb = 0.2126, 0.0722
	}
	kg := 1 - kr - kb
	ys, cs, off := 1.0, 1.0, 0.0
	if limited {
		ys, cs, off = 255.0/219, 255.0/224, 16
	}
	const one = 1 << 16
	fix := func(v float64) int32 {
		if v < 0 {
			return int32(v*one - 0.5)
		}
		return int32(v*one + 0.5)
	}
	for i := range 256 {
		c := (float64(i) - 128) * cs
		p.yt[i] = fix((float64(i) - off) * ys)
		p.rv[i] = fix(2 * (1 - kr) * c)
		p.bu[i] = fix(2 * (1 - kb) * c)
		p.gu[i] = fix(2 * (1 - kb) * kb / kg * c)
		p.gv[i] = fix(2 * (1 - kr) * kr / kg * c)
	}
}

// Invalidate says dst was written by something else: the next Process writes
// the letterbox border again.
func (p *Preprocessor) Invalidate() { p.padDone = nil }

// Process fills dst, of 3×width×height values, with src letterboxed. The
// return value says how. src is only read during the call.
func (p *Preprocessor) Process(dst []float32, src *Planes) Letterbox {
	lb := ComputeLetterbox(src.Width, src.Height, p.w, p.h)
	if lb != p.lb {
		p.lb = lb
		p.padDone = nil
		cw, ch := (src.Width+1)/2, (src.Height+1)/2
		p.sy.init(src.Width, src.Height, lb.ContentW, lb.ContentH)
		p.su.init(cw, ch, lb.ContentW, lb.ContentH)
	}
	p.setColor(src.Limited, src.BT709)
	p.fillPad(dst, lb)
	p.src, p.dst = src, dst
	parallelRows(p.workers, lb.ContentH, p.row, &p.wg)
	p.src, p.dst = nil, nil
	return lb
}

// fillPad writes the border, unless dst already has it: the rows above and
// below the picture, and the columns beside it. The picture's own pixels
// are written by every Process.
func (p *Preprocessor) fillPad(dst []float32, lb Letterbox) {
	if p.padDone == &dst[0] {
		return
	}
	ch := p.w * p.h
	top, bottom := lb.PadY*p.w, (lb.PadY+lb.ContentH)*p.w
	right := lb.PadX + lb.ContentW
	for c := range 3 {
		plane := dst[c*ch : (c+1)*ch]
		v := p.pad[c]
		fill(plane[:top], v)
		fill(plane[bottom:], v)
		if lb.PadX > 0 || right < p.w {
			for o := top; o < bottom; o += p.w {
				fill(plane[o:o+lb.PadX], v)
				fill(plane[o+right:o+p.w], v)
			}
		}
	}
	p.padDone = &dst[0]
}

func fill(s []float32, v float32) {
	for i := range s {
		s[i] = v
	}
}

// convertRows produces model rows lo..hi of the current job.
func (p *Preprocessor) convertRows(worker, lo, hi int) {
	s, lb := p.src, &p.lb
	n := lb.ContentW
	buf := p.rowPools[worker]
	yr, ur, vr := buf[:n:n], buf[n:2*n:2*n], buf[2*n:3*n:3*n]
	plane := p.w * p.h
	for row := lo; row < hi; row++ {
		p.sy.row(yr, s.Y, s.YStride, 1, row)
		if s.V == nil {
			p.su.row(ur, s.U, s.UVStride, 2, row)
			p.su.row(vr, s.U[1:], s.UVStride, 2, row)
		} else {
			p.su.row(ur, s.U, s.UVStride, 1, row)
			p.su.row(vr, s.V, s.UVStride, 1, row)
		}
		o := (row+lb.PadY)*p.w + lb.PadX
		dr := p.dst[o : o+n : o+n]
		dg := p.dst[plane+o : plane+o+n : plane+o+n]
		db := p.dst[2*plane+o : 2*plane+o+n : 2*plane+o+n]
		for x := range n {
			yv := p.yt[yr[x]]
			u, v := ur[x], vr[x]
			dr[x] = p.r[((yv+p.rv[v])>>16+clampOffset)&1023]
			dg[x] = p.g[((yv-p.gu[u]-p.gv[v])>>16+clampOffset)&1023]
			db[x] = p.b[((yv+p.bu[u])>>16+clampOffset)&1023]
		}
	}
}

// sampler scales one plane, a row at a time. Below a 1.5x reduction it
// interpolates bilinearly; beyond, it averages a window of up to 4×4 samples
// around each target pixel, which is what keeps fine detail from aliasing.
type sampler struct {
	srcW, srcH, dstW, dstH int
	copyRows               bool // same size
	kx, ky                 int  // window, 0 for bilinear
	xs                     []int32
	xf                     []uint16
	ys                     []int32
	yf                     []uint16
}

func (s *sampler) init(srcW, srcH, dstW, dstH int) {
	s.srcW, s.srcH, s.dstW, s.dstH = srcW, srcH, dstW, dstH
	s.copyRows = srcW == dstW && srcH == dstH
	s.xs, s.xf, s.kx = axis(s.xs, s.xf, srcW, dstW)
	s.ys, s.yf, s.ky = axis(s.ys, s.yf, srcH, dstH)
}

// axis returns, per target position, the first source sample and either the
// bilinear weight of the next one (k 0) or nothing (k samples are averaged).
func axis(xs []int32, xf []uint16, src, dst int) ([]int32, []uint16, int) {
	if cap(xs) < dst {
		xs = make([]int32, dst)
		xf = make([]uint16, dst)
	}
	xs, xf = xs[:dst], xf[:dst]
	ratio := float64(src) / float64(dst)
	k := 0
	if ratio >= 1.5 {
		k = min(int(ratio+0.5), 4)
	}
	for i := range dst {
		c := (float64(i) + 0.5) * ratio
		if k == 0 {
			f := c - 0.5
			if f < 0 {
				f = 0
			}
			j := int(f)
			if j >= src-1 {
				xs[i], xf[i] = int32(src-1), 0
				continue
			}
			xs[i], xf[i] = int32(j), uint16((f-float64(j))*256)
			continue
		}
		step := ratio / float64(k)
		j := int(c - step*float64(k)/2 + step/2)
		xs[i], xf[i] = int32(max(0, min(j, src-1))), uint16(max(1, int(step)))
	}
	return xs, xf, k
}

// row writes target row y to dst. stride and step locate samples: sample x of
// row r is at src[r*stride+x*step].
func (s *sampler) row(dst, src []byte, stride, step, y int) {
	n := s.dstW
	if s.copyRows {
		r := src[y*stride:]
		if step == 1 {
			copy(dst[:n], r)
			return
		}
		r = r[:n*2-1]
		for x := range n {
			dst[x] = r[x*2]
		}
		return
	}
	if s.ky == 0 {
		y0 := int(s.ys[y])
		y1 := min(y0+1, s.srcH-1)
		w1 := uint32(s.yf[y])
		w0 := 256 - w1
		r0, r1 := src[y0*stride:], src[y1*stride:]
		if s.kx == 0 {
			for x := range n {
				i := int(s.xs[x])
				j := min(i+1, s.srcW-1)
				a := uint32(s.xf[x])
				b := 256 - a
				t := uint32(r0[i*step])*b + uint32(r0[j*step])*a
				u := uint32(r1[i*step])*b + uint32(r1[j*step])*a
				dst[x] = uint8((t*w0 + u*w1 + 1<<15) >> 16)
			}
			return
		}
		k := s.kx
		for x := range n {
			i := int(s.xs[x])
			st := int(s.xf[x])
			var t, u uint32
			for c := range k {
				j := min(i+c*st, s.srcW-1) * step
				t += uint32(r0[j])
				u += uint32(r1[j])
			}
			dst[x] = uint8(((t*w0+u*w1)/uint32(k) + 1<<7) >> 8)
		}
		return
	}
	ky, ystep := s.ky, int(s.yf[y])
	y0 := int(s.ys[y])
	if s.kx == 0 {
		for x := range n {
			i := int(s.xs[x])
			j := min(i+1, s.srcW-1)
			a := uint32(s.xf[x])
			b := 256 - a
			var t uint32
			for c := range ky {
				r := src[min(y0+c*ystep, s.srcH-1)*stride:]
				t += uint32(r[i*step])*b + uint32(r[j*step])*a
			}
			dst[x] = uint8((t/uint32(ky) + 1<<7) >> 8)
		}
		return
	}
	var rows [4][]byte
	for c := range ky {
		rows[c] = src[min(y0+c*ystep, s.srcH-1)*stride:]
	}
	kx := s.kx
	total := uint32(kx * ky)
	round := total / 2
	switch {
	case kx == 2 && ky == 2:
		r0, r1 := rows[0], rows[1]
		for x := range n {
			i := int(s.xs[x])
			st := int(s.xf[x])
			j := min(i+st, s.srcW-1)
			dst[x] = uint8((uint32(r0[i*step]) + uint32(r0[j*step]) + uint32(r1[i*step]) + uint32(r1[j*step]) + 2) >> 2)
		}
	case kx == 4 && ky == 4:
		r0, r1, r2, r3 := rows[0], rows[1], rows[2], rows[3]
		for x := range n {
			st := int(s.xf[x]) * step
			i0 := int(s.xs[x]) * step
			last := (s.srcW - 1) * step
			i1, i2, i3 := min(i0+st, last), min(i0+2*st, last), min(i0+3*st, last)
			t := uint32(r0[i0]) + uint32(r0[i1]) + uint32(r0[i2]) + uint32(r0[i3]) +
				uint32(r1[i0]) + uint32(r1[i1]) + uint32(r1[i2]) + uint32(r1[i3]) +
				uint32(r2[i0]) + uint32(r2[i1]) + uint32(r2[i2]) + uint32(r2[i3]) +
				uint32(r3[i0]) + uint32(r3[i1]) + uint32(r3[i2]) + uint32(r3[i3])
			dst[x] = uint8((t + 8) >> 4)
		}
	default:
		for x := range n {
			st := int(s.xf[x])
			i := int(s.xs[x])
			var t uint32
			for c := range ky {
				r := rows[c]
				for d := range kx {
					t += uint32(r[min(i+d*st, s.srcW-1)*step])
				}
			}
			dst[x] = uint8((t + round) / total)
		}
	}
}

// --- a small shared worker pool ---

type rowJob struct {
	fn     func(worker, lo, hi int)
	worker int
	lo, hi int
	wg     *sync.WaitGroup
}

var (
	rowJobs     chan rowJob
	rowJobsOnce sync.Once
)

func startRowWorkers() {
	n := max(1, min(8, runtime.GOMAXPROCS(0)))
	rowJobs = make(chan rowJob, n*4)
	for range n {
		go func() {
			for j := range rowJobs {
				j.fn(j.worker, j.lo, j.hi)
				j.wg.Done()
			}
		}()
	}
}

// parallelRows runs fn over rows 0..rows split between workers, the caller
// doing one share. It does not allocate.
func parallelRows(workers, rows int, fn func(worker, lo, hi int), wg *sync.WaitGroup) {
	if workers <= 1 || rows < 2*workers {
		fn(0, 0, rows)
		return
	}
	rowJobsOnce.Do(startRowWorkers)
	wg.Add(workers - 1)
	for w := 1; w < workers; w++ {
		rowJobs <- rowJob{fn: fn, worker: w, lo: w * rows / workers, hi: (w + 1) * rows / workers, wg: wg}
	}
	fn(0, 0, rows/workers)
	wg.Wait()
}
