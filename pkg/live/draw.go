package live

import (
	"math"
	"strconv"

	"github.com/aep/gaal"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"golang.org/x/image/font/basicfont"
)

// Item is one thing to draw, in coordinates of the result it came from.
type Item struct {
	Label          string
	Class          int32
	Confidence     float32
	X1, Y1, X2, Y2 float32
	Mask           *gomaxv1.Mask // nil for none
}

// Mapping converts result coordinates to frame pixels.
type Mapping struct {
	SX, SY float64 // frame pixels per result coordinate
}

// NewMapping maps a coordW×coordH result space onto a frameW×frameH frame.
func NewMapping(coordW, coordH float64, frameW, frameH int) Mapping {
	if coordW <= 0 || coordH <= 0 {
		return Mapping{SX: 1, SY: 1}
	}
	return Mapping{SX: float64(frameW) / coordW, SY: float64(frameH) / coordH}
}

// Box maps (x1,y1)-(x2,y2) to integer frame coordinates, x2 and y2 exclusive,
// clamped to the frame. ok is false if nothing of it is inside.
func (m Mapping) Box(x1, y1, x2, y2 float32, frameW, frameH int) (ix1, iy1, ix2, iy2 int, ok bool) {
	ix1 = clamp(int(math.Floor(float64(x1)*m.SX+0.5)), 0, frameW)
	iy1 = clamp(int(math.Floor(float64(y1)*m.SY+0.5)), 0, frameH)
	ix2 = clamp(int(math.Floor(float64(x2)*m.SX+0.5)), 0, frameW)
	iy2 = clamp(int(math.Floor(float64(y2)*m.SY+0.5)), 0, frameH)
	return ix1, iy1, ix2, iy2, ix2 > ix1 && iy2 > iy1
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Color is a color in a frame's sample values.
type Color struct{ Y, U, V uint8 }

// RGBColor converts 8-bit RGB for frames of the given range and matrix.
func RGBColor(r, g, b uint8, full, bt709 bool) Color {
	kr, kb := 0.299, 0.114
	if bt709 {
		kr, kb = 0.2126, 0.0722
	}
	kg := 1 - kr - kb
	rf, gf, bf := float64(r)/255, float64(g)/255, float64(b)/255
	y := kr*rf + kg*gf + kb*bf
	cb := (bf - y) / (2 * (1 - kb))
	cr := (rf - y) / (2 * (1 - kr))
	var Y, U, V float64
	if full {
		Y, U, V = y*255, cb*255+128, cr*255+128
	} else {
		Y, U, V = 16+y*219, cb*224+128, cr*224+128
	}
	return Color{to8(Y), to8(U), to8(V)}
}

func to8(v float64) uint8 {
	switch {
	case v <= 0:
		return 0
	case v >= 255:
		return 255
	}
	return uint8(v + 0.5)
}

// palette gives each class a stable, distinct color.
var palette = [...][3]uint8{
	{255, 56, 56}, {56, 200, 255}, {56, 255, 96}, {255, 178, 29}, {207, 56, 255},
	{255, 112, 180}, {0, 255, 200}, {180, 220, 40}, {80, 120, 255}, {255, 140, 60},
	{120, 255, 180}, {200, 200, 255},
}

// Overlay draws results into frames. It is for one goroutine and reuses its
// buffers, so drawing does not allocate once sizes are stable.
type Overlay struct {
	colors  [len(palette)]Color
	full    bool
	bt709   bool
	ready   bool
	text    []byte
	xMap    []int32
	glyphs  [95][13]uint8
	loaded  bool
	white   Color
	textDim Color
}

// colorsFor prepares the palette for a frame's color description.
func (o *Overlay) colorsFor(full, bt709 bool) {
	if o.ready && o.full == full && o.bt709 == bt709 {
		return
	}
	o.full, o.bt709, o.ready = full, bt709, true
	for i, c := range palette {
		o.colors[i] = RGBColor(c[0], c[1], c[2], full, bt709)
	}
	o.white = RGBColor(255, 255, 255, full, bt709)
	o.textDim = RGBColor(0, 0, 0, full, bt709)
}

// loadFont reads the ASCII glyphs of the 7×13 bitmap font.
func (o *Overlay) loadFont() {
	if o.loaded {
		return
	}
	o.loaded = true
	f := basicfont.Face7x13
	stride := f.Mask.Bounds().Dx()
	for c := 32; c < 127; c++ {
		rng := f.Ranges[0]
		idx := c - int(rng.Low) + rng.Offset
		_ = stride
		for row := 0; row < 13; row++ {
			var bits uint8
			for x := 0; x < 7; x++ {
				_, _, _, a := f.Mask.At(x, idx*13+row).RGBA()
				if a > 0x7fff {
					bits |= 1 << x
				}
			}
			o.glyphs[c-32][row] = bits
		}
	}
}

// Draw renders items into f, whose pixels it changes. coordW and coordH are
// the result's coordinate space. bt709 is the frame's matrix.
func (o *Overlay) Draw(f *gaal.VideoFrame, items []Item, coordW, coordH float64, bt709 bool) {
	o.colorsFor(f.Range == gaal.RangeFull, bt709)
	o.loadFont()
	m := NewMapping(coordW, coordH, f.Width, f.Height)
	thick := max(2, f.Height/360)
	scale := max(1, f.Height/480)
	banner := 0
	for i := range items {
		it := &items[i]
		c := o.colors[int(uint32(it.Class))%len(o.colors)]
		if it.Mask != nil {
			o.drawMask(f, it.Mask, m, c)
		}
		if it.X2 <= it.X1 || it.Y2 <= it.Y1 {
			// A classification: stack labels in the corner.
			o.drawLabel(f, it, 4, 4+banner, scale, c)
			banner += 15 * scale
			continue
		}
		x1, y1, x2, y2, ok := m.Box(it.X1, it.Y1, it.X2, it.Y2, f.Width, f.Height)
		if !ok {
			continue
		}
		o.rect(f, x1, y1, x2, y2, thick, c)
		ly := y1 - 15*scale
		if ly < 0 {
			ly = y1
		}
		o.drawLabel(f, it, x1, ly, scale, c)
	}
}

func (o *Overlay) rect(f *gaal.VideoFrame, x1, y1, x2, y2, t int, c Color) {
	fill(f, x1, y1, x2, min(y1+t, y2), c)
	fill(f, x1, max(y2-t, y1), x2, y2, c)
	fill(f, x1, y1, min(x1+t, x2), y2, c)
	fill(f, max(x2-t, x1), y1, x2, y2, c)
}

// fill paints the pixels x0..x1, y0..y1 (exclusive ends, clamped).
func fill(f *gaal.VideoFrame, x0, y0, x1, y1 int, c Color) {
	x0, x1 = clamp(x0, 0, f.Width), clamp(x1, 0, f.Width)
	y0, y1 = clamp(y0, 0, f.Height), clamp(y1, 0, f.Height)
	if x1 <= x0 || y1 <= y0 {
		return
	}
	for y := y0; y < y1; y++ {
		row := f.Y[y*f.YStride+x0 : y*f.YStride+x1]
		for i := range row {
			row[i] = c.Y
		}
	}
	cx0, cx1 := x0/2, (x1+1)/2
	for cy := y0 / 2; cy < (y1+1)/2; cy++ {
		if f.Format == gaal.NV12 {
			row := f.U[cy*f.UVStride+cx0*2 : cy*f.UVStride+cx1*2]
			for i := 0; i+1 < len(row); i += 2 {
				row[i], row[i+1] = c.U, c.V
			}
			continue
		}
		ur := f.U[cy*f.UVStride+cx0 : cy*f.UVStride+cx1]
		vr := f.V[cy*f.UVStride+cx0 : cy*f.UVStride+cx1]
		for i := range ur {
			ur[i], vr[i] = c.U, c.V
		}
	}
}

// drawLabel writes "label 87%" with a background of color c at (x, y).
func (o *Overlay) drawLabel(f *gaal.VideoFrame, it *Item, x, y, scale int, c Color) {
	t := append(o.text[:0], it.Label...)
	if it.Confidence > 0 {
		if len(t) > 0 {
			t = append(t, ' ')
		}
		t = strconv.AppendInt(t, int64(it.Confidence*100+0.5), 10)
		t = append(t, '%')
	}
	o.text = t
	if len(t) == 0 {
		return
	}
	adv, h := 7*scale, 13*scale
	w := min(len(t)*adv+4*scale, f.Width-x)
	fill(f, x, y, x+w, y+h+2*scale, c)
	white := o.white.Y
	if int(c.Y) > (int(o.white.Y)+int(o.textDim.Y))*6/10 {
		white = o.textDim.Y // dark text on a bright label
	}
	px := x + 2*scale
	for _, ch := range t {
		if px+adv > f.Width {
			break
		}
		if ch < 32 || ch >= 127 {
			ch = '?'
		}
		g := &o.glyphs[ch-32]
		for row, bits := range g {
			for sy := 0; sy < scale; sy++ {
				py := y + scale + row*scale + sy
				if py < 0 || py >= f.Height || bits == 0 {
					continue
				}
				r := f.Y[py*f.YStride : py*f.YStride+f.Width]
				for bx := 0; bx < 7; bx++ {
					if bits&(1<<bx) == 0 {
						continue
					}
					for sx := 0; sx < scale; sx++ {
						if q := px + bx*scale + sx; q < len(r) {
							r[q] = white
						}
					}
				}
			}
		}
		px += adv
	}
}

// drawMask blends c over the mask's pixels, one half each.
func (o *Overlay) drawMask(f *gaal.VideoFrame, mk *gomaxv1.Mask, m Mapping, c Color) {
	mw, mh := int(mk.GetWidth()), int(mk.GetHeight())
	bits := mk.GetBits()
	if mw <= 0 || mh <= 0 || len(bits)*8 < mw*mh || m.SX <= 0 || m.SY <= 0 {
		return
	}
	mx, my := float64(mk.GetX()), float64(mk.GetY())
	fx0 := clamp(int(math.Floor(mx*m.SX)), 0, f.Width)
	fy0 := clamp(int(math.Floor(my*m.SY)), 0, f.Height)
	fx1 := clamp(int(math.Ceil((mx+float64(mw))*m.SX)), 0, f.Width)
	fy1 := clamp(int(math.Ceil((my+float64(mh))*m.SY)), 0, f.Height)
	if fx1 <= fx0 || fy1 <= fy0 {
		return
	}
	if cap(o.xMap) < fx1-fx0 {
		o.xMap = make([]int32, f.Width)
	}
	xmap := o.xMap[:fx1-fx0]
	for i := range xmap {
		v := int(math.Floor((float64(fx0+i)+0.5)/m.SX - mx))
		if v < 0 || v >= mw {
			v = -1
		}
		xmap[i] = int32(v)
	}
	nv12 := f.Format == gaal.NV12
	cy, cu, cv := uint16(c.Y), uint16(c.U), uint16(c.V)
	for y := fy0; y < fy1; y++ {
		v := int(math.Floor((float64(y)+0.5)/m.SY - my))
		if v < 0 || v >= mh {
			continue
		}
		base := v * mw
		yr := f.Y[y*f.YStride+fx0 : y*f.YStride+fx1]
		for i, mxv := range xmap {
			if mxv < 0 {
				continue
			}
			b := base + int(mxv)
			if bits[b>>3]&(0x80>>(b&7)) != 0 {
				yr[i] = uint8((uint16(yr[i]) + cy + 1) >> 1)
			}
		}
		if y&1 != 0 {
			continue
		}
		cRow := (y / 2) * f.UVStride
		for i, mxv := range xmap {
			x := fx0 + i
			if x&1 != 0 || mxv < 0 {
				continue
			}
			b := base + int(mxv)
			if bits[b>>3]&(0x80>>(b&7)) == 0 {
				continue
			}
			if nv12 {
				p := cRow + x
				f.U[p] = uint8((uint16(f.U[p]) + cu + 1) >> 1)
				f.U[p+1] = uint8((uint16(f.U[p+1]) + cv + 1) >> 1)
			} else {
				p := cRow + x/2
				f.U[p] = uint8((uint16(f.U[p]) + cu + 1) >> 1)
				f.V[p] = uint8((uint16(f.V[p]) + cv + 1) >> 1)
			}
		}
	}
}
