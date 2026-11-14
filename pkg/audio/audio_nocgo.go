//go:build !cgo

package audio

import "errors"

// errNoCgo: the MP3 and FLAC decoders are C (minimp3, dr_flac); a build
// without cgo decodes WAV only.
var errNoCgo = errors.New("audio: MP3 and FLAC decoding needs a cgo build of moxgo; use WAV")

func decodeMP3([]byte) ([]float32, int, error) { return nil, 0, errNoCgo }

func decodeFLAC([]byte) ([]float32, int, error) { return nil, 0, errNoCgo }
