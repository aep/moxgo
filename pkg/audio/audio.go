// Package audio decodes audio files and prepares them as float32 tensors
// for direct use with pkg/onnx.
//
// Supported formats: WAV, MP3, FLAC.
// Resamples to target rate, mixes to mono, normalizes to [-1, 1].
package audio

/*
#include <stdlib.h>

struct mp3_result {
    float *samples;
    int n_samples;
    int channels;
    int sample_rate;
};
struct mp3_result mp3_decode(const unsigned char *data, int data_len);
void mp3_free(struct mp3_result *r);

struct flac_result {
    float *samples;
    int n_samples;
    int channels;
    int sample_rate;
};
struct flac_result flac_decode(const unsigned char *data, int data_len);
void flac_free(struct flac_result *r);
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"unsafe"
)

// Options configures audio decoding for a target model.
type Options struct {
	SampleRate int // Target sample rate. 0 = keep original.
	MaxSamples int // Max output samples. 0 = no limit. Pads with silence or truncates.
}

// Tensor holds decoded audio ready for ONNX inference.
type Tensor struct {
	Data       []float32
	Shape      []int64 // [1, samples]
	SampleRate int
	Duration   float64 // seconds
}

// Decode reads audio from r, resamples and normalizes to [-1, 1] float32 mono.
// Detects format by magic bytes (WAV, MP3, FLAC).
func Decode(r io.Reader, opts Options) (*Tensor, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("audio: read: %w", err)
	}
	if len(data) < 4 {
		return nil, fmt.Errorf("audio: file too short")
	}

	var samples []float32
	var srcRate int

	switch {
	case string(data[:4]) == "RIFF":
		samples, srcRate, err = decodeWAVBytes(data)
	case string(data[:4]) == "fLaC":
		samples, srcRate, err = decodeFLAC(data)
	case isMP3(data):
		samples, srcRate, err = decodeMP3(data)
	default:
		return nil, fmt.Errorf("audio: unknown format (magic: %x)", data[:4])
	}
	if err != nil {
		return nil, err
	}

	// Resample if needed
	targetRate := opts.SampleRate
	if targetRate == 0 {
		targetRate = srcRate
	}
	if targetRate != srcRate {
		samples = resample(samples, srcRate, targetRate)
	}

	// Pad or truncate
	if opts.MaxSamples > 0 {
		if len(samples) > opts.MaxSamples {
			samples = samples[:opts.MaxSamples]
		} else if len(samples) < opts.MaxSamples {
			pad := make([]float32, opts.MaxSamples-len(samples))
			samples = append(samples, pad...)
		}
	}

	return &Tensor{
		Data:       samples,
		Shape:      []int64{1, int64(len(samples))},
		SampleRate: targetRate,
		Duration:   float64(len(samples)) / float64(targetRate),
	}, nil
}

// isMP3 detects MP3 by frame sync or ID3 tag.
func isMP3(data []byte) bool {
	if len(data) >= 3 && string(data[:3]) == "ID3" {
		return true
	}
	// MP3 frame sync: 11 set bits
	if len(data) >= 2 && data[0] == 0xFF && (data[1]&0xE0) == 0xE0 {
		return true
	}
	return false
}

func decodeMP3(data []byte) ([]float32, int, error) {
	res := C.mp3_decode((*C.uchar)(unsafe.Pointer(&data[0])), C.int(len(data)))
	if res.samples == nil || res.n_samples == 0 {
		return nil, 0, fmt.Errorf("audio: mp3 decode failed")
	}
	defer C.mp3_free(&res)

	channels := int(res.channels)
	sampleRate := int(res.sample_rate)
	totalSamples := int(res.n_samples)
	nFrames := totalSamples / channels

	// Copy and mix to mono
	cPtr := (*[1 << 30]float32)(unsafe.Pointer(res.samples))
	mono := make([]float32, nFrames)
	if channels == 1 {
		copy(mono, cPtr[:nFrames])
	} else {
		for i := 0; i < nFrames; i++ {
			var sum float32
			for ch := 0; ch < channels; ch++ {
				sum += cPtr[i*channels+ch]
			}
			mono[i] = sum / float32(channels)
		}
	}
	return mono, sampleRate, nil
}

func decodeFLAC(data []byte) ([]float32, int, error) {
	res := C.flac_decode((*C.uchar)(unsafe.Pointer(&data[0])), C.int(len(data)))
	if res.samples == nil || res.n_samples == 0 {
		return nil, 0, fmt.Errorf("audio: flac decode failed")
	}
	defer C.flac_free(&res)

	channels := int(res.channels)
	sampleRate := int(res.sample_rate)
	totalSamples := int(res.n_samples)
	nFrames := totalSamples / channels

	cPtr := (*[1 << 30]float32)(unsafe.Pointer(res.samples))
	mono := make([]float32, nFrames)
	if channels == 1 {
		copy(mono, cPtr[:nFrames])
	} else {
		for i := 0; i < nFrames; i++ {
			var sum float32
			for ch := 0; ch < channels; ch++ {
				sum += cPtr[i*channels+ch]
			}
			mono[i] = sum / float32(channels)
		}
	}
	return mono, sampleRate, nil
}

// decodeWAVBytes parses WAV from a byte slice.
func decodeWAVBytes(data []byte) ([]float32, int, error) {
	if len(data) < 44 {
		return nil, 0, fmt.Errorf("audio: WAV too short")
	}
	if string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("audio: not WAVE")
	}

	var (
		audioFmt   uint16
		channels   uint16
		sampleRate uint32
		bitsPerSmp uint16
		dataBytes  []byte
	)

	pos := 12
	for pos+8 <= len(data) {
		chunkID := string(data[pos : pos+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		pos += 8
		if pos+chunkSize > len(data) {
			chunkSize = len(data) - pos
		}

		switch chunkID {
		case "fmt ":
			if chunkSize < 16 {
				return nil, 0, fmt.Errorf("audio: fmt chunk too short")
			}
			audioFmt = binary.LittleEndian.Uint16(data[pos : pos+2])
			channels = binary.LittleEndian.Uint16(data[pos+2 : pos+4])
			sampleRate = binary.LittleEndian.Uint32(data[pos+4 : pos+8])
			bitsPerSmp = binary.LittleEndian.Uint16(data[pos+14 : pos+16])
		case "data":
			dataBytes = data[pos : pos+chunkSize]
		}
		pos += chunkSize
		if pos%2 != 0 {
			pos++ // chunks are word-aligned
		}
	}

	if dataBytes == nil {
		return nil, 0, fmt.Errorf("audio: no data chunk")
	}
	if channels == 0 || sampleRate == 0 {
		return nil, 0, fmt.Errorf("audio: missing fmt chunk")
	}

	bytesPerSample := int(bitsPerSmp) / 8
	frameSize := bytesPerSample * int(channels)
	nFrames := len(dataBytes) / frameSize
	samples := make([]float32, nFrames)

	switch {
	case audioFmt == 1 && bitsPerSmp == 8:
		for i := 0; i < nFrames; i++ {
			var sum float32
			for ch := 0; ch < int(channels); ch++ {
				v := dataBytes[i*frameSize+ch]
				sum += (float32(v) - 128) / 128.0
			}
			samples[i] = sum / float32(channels)
		}

	case audioFmt == 1 && bitsPerSmp == 16:
		for i := 0; i < nFrames; i++ {
			var sum float32
			for ch := 0; ch < int(channels); ch++ {
				off := i*frameSize + ch*2
				v := int16(binary.LittleEndian.Uint16(dataBytes[off : off+2]))
				sum += float32(v) / 32768.0
			}
			samples[i] = sum / float32(channels)
		}

	case audioFmt == 1 && bitsPerSmp == 24:
		for i := 0; i < nFrames; i++ {
			var sum float32
			for ch := 0; ch < int(channels); ch++ {
				off := i*frameSize + ch*3
				v := int32(dataBytes[off]) | int32(dataBytes[off+1])<<8 | int32(dataBytes[off+2])<<16
				if v&0x800000 != 0 {
					v |= ^0xFFFFFF
				}
				sum += float32(v) / 8388608.0
			}
			samples[i] = sum / float32(channels)
		}

	case audioFmt == 1 && bitsPerSmp == 32:
		for i := 0; i < nFrames; i++ {
			var sum float32
			for ch := 0; ch < int(channels); ch++ {
				off := i*frameSize + ch*4
				v := int32(binary.LittleEndian.Uint32(dataBytes[off : off+4]))
				sum += float32(v) / 2147483648.0
			}
			samples[i] = sum / float32(channels)
		}

	case audioFmt == 3 && bitsPerSmp == 32:
		for i := 0; i < nFrames; i++ {
			var sum float32
			for ch := 0; ch < int(channels); ch++ {
				off := i*frameSize + ch*4
				bits := binary.LittleEndian.Uint32(dataBytes[off : off+4])
				sum += math.Float32frombits(bits)
			}
			samples[i] = sum / float32(channels)
		}

	default:
		return nil, 0, fmt.Errorf("audio: unsupported WAV format (fmt=%d, bits=%d)", audioFmt, bitsPerSmp)
	}

	return samples, int(sampleRate), nil
}

// resample using linear interpolation.
func resample(in []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate || len(in) == 0 {
		return in
	}
	ratio := float64(srcRate) / float64(dstRate)
	outLen := int(float64(len(in)) / ratio)
	out := make([]float32, outLen)
	for i := range out {
		srcIdx := float64(i) * ratio
		idx := int(srcIdx)
		frac := float32(srcIdx - float64(idx))
		if idx+1 < len(in) {
			out[i] = in[idx]*(1-frac) + in[idx+1]*frac
		} else if idx < len(in) {
			out[i] = in[idx]
		}
	}
	return out
}
