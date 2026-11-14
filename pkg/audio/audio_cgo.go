//go:build cgo

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
	"fmt"
	"unsafe"
)

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
