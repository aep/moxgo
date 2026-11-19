package camera

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aep/gaal"
)

func TestRawToVideoFrame(t *testing.T) {
	var pool gaal.FramePool
	// 4×2 YUYV: chroma of the two rows is averaged into one 4:2:0 row.
	yuyv := []byte{
		10, 100, 11, 200, 12, 110, 13, 210,
		20, 120, 21, 220, 22, 130, 23, 230,
	}
	f, err := rawToVideoFrame(&rawFrame{format: pixFmtYUYV, width: 4, height: 2, planes: [][]byte{yuyv}}, &pool)
	if err != nil {
		t.Fatal(err)
	}
	if f.Format != gaal.I420 || f.Y[1] != 11 || f.Y[f.YStride] != 20 || f.U[0] != 110 || f.U[1] != 120 || f.V[0] != 210 {
		t.Fatalf("got Y=%v U=%v V=%v", f.Y[:8], f.U[:2], f.V[:2])
	}
	f.Release()

	// NV21 with padding: U and V swapped into NV12.
	nv21 := []byte{
		1, 2, 9, 9, 3, 4, 9, 9, // luma, stride 4
		50, 60, 9, 9, // VU
	}
	f, err = rawToVideoFrame(&rawFrame{format: pixFmtNV21, width: 2, height: 2, strides: []int{4}, planes: [][]byte{nv21}}, &pool)
	if err != nil {
		t.Fatal(err)
	}
	if f.Format != gaal.NV12 || f.Y[f.YStride+1] != 4 || f.U[0] != 60 || f.U[1] != 50 {
		t.Fatalf("got %v %v", f.Y[:4], f.U[:2])
	}
	f.Release()

	if _, err := rawToVideoFrame(&rawFrame{format: pixFmtYUYV, width: 4, height: 4, planes: [][]byte{yuyv}}, &pool); err == nil {
		t.Fatal("short frame accepted")
	}
}

func TestImageToFrameAndBack(t *testing.T) {
	var pool gaal.FramePool
	img := image.NewYCbCr(image.Rect(0, 0, 4, 4), image.YCbCrSubsampleRatio420)
	for i := range img.Y {
		img.Y[i] = byte(i)
	}
	img.Cb[3], img.Cr[3] = 77, 88
	f := ImageToFrame(img, &pool)
	defer f.Release()
	if f.Range != gaal.RangeFull || f.U[3] != 77 || f.V[3] != 88 || f.Y[5] != 5 {
		t.Fatalf("frame %+v", f.U)
	}
	back := FrameToImage(f)
	if back.Y[5] != 5 || back.Cb[3] != 77 || back.Cr[3] != 88 {
		t.Fatal("round trip changed samples")
	}
}

func TestRangeProbe(t *testing.T) {
	limited := gaal.NewVideoFrame(64, 64, gaal.NV12)
	for i := range limited.Y {
		limited.Y[i] = 16 + byte(i%219)
	}
	var p rangeProbe
	p.apply(limited)
	if limited.Range != gaal.RangeLimited {
		t.Fatalf("limited picture read as %v", limited.Range)
	}
	full := gaal.NewVideoFrame(64, 64, gaal.NV12)
	for i := range full.Y {
		full.Y[i] = byte(i)
	}
	p.apply(full)
	if full.Range != gaal.RangeFull {
		t.Fatalf("full picture read as %v", full.Range)
	}
	// Decided once: later frames stay full range even if dark.
	again := gaal.NewVideoFrame(64, 64, gaal.NV12)
	for i := range again.Y {
		again.Y[i] = 100
	}
	p.apply(again)
	if again.Range != gaal.RangeFull {
		t.Fatalf("got %v", again.Range)
	}
	// A stated range is left alone.
	stated := gaal.NewVideoFrame(64, 64, gaal.NV12)
	stated.Range = gaal.RangeLimited
	p.apply(stated)
	if stated.Range != gaal.RangeLimited {
		t.Fatal("stated range changed")
	}
}

func TestHTTPMJPEGSource(t *testing.T) {
	var jpg bytes.Buffer
	img := image.NewYCbCr(image.Rect(0, 0, 32, 24), image.YCbCrSubsampleRatio420)
	for i := range img.Y {
		img.Y[i] = 200
	}
	for i := range img.Cb {
		img.Cb[i], img.Cr[i] = 128, 128
	}
	if err := jpeg.Encode(&jpg, img, nil); err != nil {
		t.Fatal(err)
	}
	gotUser := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u, _, _ := r.BasicAuth()
		select {
		case gotUser <- u:
		default:
		}
		w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
		for i := 0; i < 3; i++ {
			fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", jpg.Len())
			w.Write(jpg.Bytes())
			fmt.Fprint(w, "\r\n")
			w.(http.Flusher).Flush()
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer srv.Close()
	u := strings.Replace(srv.URL, "http://", "http://viewer:pw@", 1)
	src, err := OpenVideoStream(u+"/video", SourceConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if strings.Contains(src.Name(), "pw") || strings.Contains(src.Name(), "viewer") {
		t.Fatalf("name leaks credentials: %s", src.Name())
	}
	f, tm, err := src.Next(3 * time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Release()
	if f.Width != 32 || f.Height != 24 || f.Range != gaal.RangeFull || tm.Decoded.IsZero() {
		t.Fatalf("frame %dx%d range %v", f.Width, f.Height, f.Range)
	}
	if y := f.Y[f.YStride*12+16]; y < 190 || y > 210 {
		t.Fatalf("luma %d", y)
	}
	if got := <-gotUser; got != "viewer" {
		t.Fatalf("basic auth user %q", got)
	}
}
