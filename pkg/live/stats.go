package live

import (
	"encoding/json"
	"fmt"
	"io"
	"runtime/metrics"
	"slices"
	"strings"
	"sync"
	"time"
)

// Stage names one measured step or latency.
type Stage int

// Stages. The first group is per video frame, the second per inference, the
// last are latencies over the whole path.
const (
	StageReceive Stage = iota // capture timestamp / network arrival to data in hand
	StageDecode               // data handed to the decoder until the frame is out
	StageCopy                 // copying a frame that inference still reads
	StageDraw                 // drawing detections
	StageShow                 // Display.Show
	StagePresent              // Show until the display reports presentation

	StageQueue       // waiting in the mailbox for inference
	StagePreprocess  // scaling and color conversion into the tensor
	StageInference   // model execution (and the RPC for a remote model)
	StagePostprocess // output decoding

	StageVideoLatency   // capture to Show returning
	StageDisplayLatency // capture to presented on screen
	StageDetLatency     // capture of a frame to its detections being available
	StageDetAge         // age of the shown detections' frame when shown

	numStages
)

var stageNames = [numStages]string{
	"receive", "decode", "copy", "draw", "show", "present",
	"queue-wait", "preprocess", "inference", "postprocess",
	"video-latency", "display-latency", "detect-latency", "detect-age",
}

// String is the stage's name in reports.
func (s Stage) String() string { return stageNames[s] }

const histSize = 1 << 12

type hist struct {
	n     uint64
	sum   time.Duration
	max   time.Duration
	ring  [histSize]float32 // ms
	used  int
	nIval uint64
	sumIv time.Duration
}

func (h *hist) add(d time.Duration) {
	h.n++
	h.sum += d
	h.nIval++
	h.sumIv += d
	if d > h.max {
		h.max = d
	}
	h.ring[h.used%histSize] = float32(d.Seconds() * 1000)
	h.used++
}

// StageSummary is a stage's figures in milliseconds.
type StageSummary struct {
	Count uint64  `json:"count"`
	Avg   float64 `json:"avg_ms"`
	P50   float64 `json:"p50_ms"`
	P95   float64 `json:"p95_ms"`
	Max   float64 `json:"max_ms"`
}

// Stats collects pipeline measurements. Its methods are safe for concurrent
// use and do not allocate.
type Stats struct {
	mu     sync.Mutex
	h      [numStages]hist
	frames uint64 // video frames through the pipeline
	shown  uint64
	dets   uint64 // inferences finished
	boxes  uint64 // detections in those
	drawn  uint64 // video frames with detections drawn
	errors uint64 // failed inferences
	copies uint64 // frames copied before drawing

	start     time.Time
	cpuStart  time.Duration
	allocs    [2]uint64
	lastPrint time.Time
	lastFr    uint64
	lastDet   uint64
	lastCPU   time.Duration
	lastAlloc [2]uint64
}

var allocSamples = []metrics.Sample{{Name: "/gc/heap/allocs:objects"}, {Name: "/gc/heap/allocs:bytes"}}

func readAllocs() [2]uint64 {
	s := make([]metrics.Sample, len(allocSamples))
	copy(s, allocSamples)
	metrics.Read(s)
	return [2]uint64{s[0].Value.Uint64(), s[1].Value.Uint64()}
}

// NewStats starts measuring.
func NewStats() *Stats {
	s := &Stats{}
	s.Reset()
	return s
}

// Reset forgets everything measured so far, e.g. after startup.
func (s *Stats) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.h = [numStages]hist{}
	s.frames, s.shown, s.dets, s.boxes, s.drawn, s.errors, s.copies = 0, 0, 0, 0, 0, 0, 0
	s.start = time.Now()
	s.cpuStart = processCPU()
	s.allocs = readAllocs()
	s.lastPrint, s.lastFr, s.lastDet, s.lastCPU, s.lastAlloc = s.start, 0, 0, s.cpuStart, s.allocs
}

// Add records a stage's duration; d < 0 is ignored.
func (s *Stats) Add(st Stage, d time.Duration) {
	if d < 0 {
		return
	}
	s.mu.Lock()
	s.h[st].add(d)
	s.mu.Unlock()
}

func (s *Stats) count(p *uint64, n uint64) {
	s.mu.Lock()
	*p += n
	s.mu.Unlock()
}

// Frame counts a video frame through the pipeline.
func (s *Stats) Frame() { s.count(&s.frames, 1) }

// Shown counts a frame accepted by the display.
func (s *Stats) Shown() { s.count(&s.shown, 1) }

// Drawn counts a frame with detections drawn into it.
func (s *Stats) Drawn() { s.count(&s.drawn, 1) }

// Copied counts a frame copied before drawing.
func (s *Stats) Copied() { s.count(&s.copies, 1) }

// Detection counts an inference and how many objects it found.
func (s *Stats) Detection(objects int) {
	s.mu.Lock()
	s.dets++
	s.boxes += uint64(objects)
	s.mu.Unlock()
}

// DetectError counts a failed inference.
func (s *Stats) DetectError() { s.count(&s.errors, 1) }

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func (h *hist) summary() StageSummary {
	if h.n == 0 {
		return StageSummary{}
	}
	n := min(h.used, histSize)
	v := append([]float32(nil), h.ring[:n]...)
	slices.Sort(v)
	return StageSummary{
		Count: h.n,
		Avg:   ms(h.sum / time.Duration(h.n)),
		P50:   float64(v[n/2]),
		P95:   float64(v[n*95/100]),
		Max:   ms(h.max),
	}
}

// Summary is what a run measured.
type Summary struct {
	Seconds       float64                 `json:"seconds"`
	VideoFPS      float64                 `json:"video_fps"`
	ShownFPS      float64                 `json:"shown_fps"`
	DetectFPS     float64                 `json:"detect_fps"`
	ObjectsPerDet float64                 `json:"objects_per_detection"`
	DrawnFrames   uint64                  `json:"frames_with_detections"`
	CopiedFrames  uint64                  `json:"frames_copied"`
	DetectErrors  uint64                  `json:"detect_errors"`
	CPUPercent    float64                 `json:"cpu_percent"` // of one core
	AllocsPerFr   float64                 `json:"allocs_per_frame"`
	AllocBytesFr  float64                 `json:"alloc_bytes_per_frame"`
	Stages        map[string]StageSummary `json:"stages"`
}

// Summary returns the figures since the last Reset.
func (s *Stats) Summary() Summary {
	s.mu.Lock()
	defer s.mu.Unlock()
	el := time.Since(s.start).Seconds()
	a := readAllocs()
	out := Summary{
		Seconds:      el,
		VideoFPS:     float64(s.frames) / el,
		ShownFPS:     float64(s.shown) / el,
		DetectFPS:    float64(s.dets) / el,
		DrawnFrames:  s.drawn,
		CopiedFrames: s.copies,
		DetectErrors: s.errors,
		CPUPercent:   100 * (processCPU() - s.cpuStart).Seconds() / el,
		Stages:       map[string]StageSummary{},
	}
	if s.dets > 0 {
		out.ObjectsPerDet = float64(s.boxes) / float64(s.dets)
	}
	if s.frames > 0 {
		out.AllocsPerFr = float64(a[0]-s.allocs[0]) / float64(s.frames)
		out.AllocBytesFr = float64(a[1]-s.allocs[1]) / float64(s.frames)
	}
	for i := range s.h {
		if s.h[i].n > 0 {
			out.Stages[Stage(i).String()] = s.h[i].summary()
		}
	}
	return out
}

// PrintInterval writes one line with the figures since the last call.
func (s *Stats) PrintInterval(w io.Writer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	el := now.Sub(s.lastPrint).Seconds()
	if el <= 0 {
		return
	}
	cpu := processCPU()
	a := readAllocs()
	frames := s.frames - s.lastFr
	var sb strings.Builder
	fmt.Fprintf(&sb, "video %5.1f fps  det %5.1f fps  cpu %4.0f%%  allocs/frame %.0f |", float64(frames)/el,
		float64(s.dets-s.lastDet)/el, 100*(cpu-s.lastCPU).Seconds()/el, float64(a[0]-s.lastAlloc[0])/max(float64(frames), 1))
	for _, st := range []Stage{StageDecode, StageDraw, StageShow, StagePreprocess, StageInference, StagePostprocess, StageVideoLatency, StageDisplayLatency, StageDetAge} {
		h := &s.h[st]
		if h.nIval > 0 {
			fmt.Fprintf(&sb, " %s %.2f", st, ms(h.sumIv/time.Duration(h.nIval)))
		}
	}
	for i := range s.h {
		s.h[i].nIval, s.h[i].sumIv = 0, 0
	}
	s.lastPrint, s.lastFr, s.lastDet, s.lastCPU, s.lastAlloc = now, s.frames, s.dets, cpu, a
	fmt.Fprintln(w, sb.String())
}

// WriteSummary prints a Summary as a table, or as JSON.
func WriteSummary(w io.Writer, sum Summary, asJSON bool) {
	if asJSON {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		_ = enc.Encode(sum)
		return
	}
	fmt.Fprintf(w, "\n%.1fs  video %.1f fps  shown %.1f fps  detection %.1f fps (%.2f objects each)  frames with detections %d\n",
		sum.Seconds, sum.VideoFPS, sum.ShownFPS, sum.DetectFPS, sum.ObjectsPerDet, sum.DrawnFrames)
	fmt.Fprintf(w, "cpu %.0f%% of a core  allocations/frame %.0f (%.0f bytes)  frames copied for drawing %d  detect errors %d\n",
		sum.CPUPercent, sum.AllocsPerFr, sum.AllocBytesFr, sum.CopiedFrames, sum.DetectErrors)
	fmt.Fprintf(w, "%-16s %8s %9s %9s %9s %9s\n", "stage (ms)", "count", "avg", "p50", "p95", "max")
	for i := Stage(0); i < numStages; i++ {
		if v, ok := sum.Stages[i.String()]; ok {
			fmt.Fprintf(w, "%-16s %8d %9.2f %9.2f %9.2f %9.2f\n", i.String(), v.Count, v.Avg, v.P50, v.P95, v.Max)
		}
	}
}
