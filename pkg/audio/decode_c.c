#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"

#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#include "dr_flac.h"

#include <stdlib.h>
#include <string.h>

// --- MP3 decoding ---

struct mp3_result {
    float *samples;    // interleaved float32 PCM
    int n_samples;     // total samples (frames * channels)
    int channels;
    int sample_rate;
};

struct mp3_result mp3_decode(const unsigned char *data, int data_len) {
    struct mp3_result res = {0};
    mp3dec_t dec;
    mp3dec_init(&dec);

    // Worst case: MP3 frame = 1152 samples. Estimate max frames.
    int alloc = 1152 * 2 * (data_len / 100 + 1);
    float *buf = (float *)malloc(alloc * sizeof(float));
    if (!buf) return res;

    int total = 0;
    int offset = 0;
    mp3dec_frame_info_t info;

    while (offset < data_len) {
        short pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
        int samples = mp3dec_decode_frame(&dec, data + offset, data_len - offset, pcm, &info);
        if (info.frame_bytes == 0) break;
        offset += info.frame_bytes;

        if (samples > 0) {
            if (res.sample_rate == 0) {
                res.sample_rate = info.hz;
                res.channels = info.channels;
            }
            int count = samples * info.channels;
            // Grow buffer if needed
            if (total + count > alloc) {
                alloc = (total + count) * 2;
                buf = (float *)realloc(buf, alloc * sizeof(float));
                if (!buf) { res.n_samples = 0; return res; }
            }
            // Convert s16 to float32
            for (int i = 0; i < count; i++) {
                buf[total + i] = pcm[i] / 32768.0f;
            }
            total += count;
        }
    }

    res.samples = buf;
    res.n_samples = total;
    return res;
}

void mp3_free(struct mp3_result *r) {
    if (r->samples) {
        free(r->samples);
        r->samples = NULL;
    }
}

// --- FLAC decoding (from memory) ---

struct flac_result {
    float *samples;    // interleaved float32 PCM
    int n_samples;     // total samples (frames * channels)
    int channels;
    int sample_rate;
};

struct flac_result flac_decode(const unsigned char *data, int data_len) {
    struct flac_result res = {0};

    drflac *flac = drflac_open_memory(data, (size_t)data_len, NULL);
    if (!flac) return res;

    res.channels = (int)flac->channels;
    res.sample_rate = (int)flac->sampleRate;

    drflac_uint64 total_frames = flac->totalPCMFrameCount;
    int total_samples = (int)(total_frames * flac->channels);

    float *buf = (float *)malloc(total_samples * sizeof(float));
    if (!buf) {
        drflac_close(flac);
        return res;
    }

    drflac_uint64 read = drflac_read_pcm_frames_f32(flac, total_frames, buf);
    res.samples = buf;
    res.n_samples = (int)(read * flac->channels);

    drflac_close(flac);
    return res;
}

void flac_free(struct flac_result *r) {
    if (r->samples) {
        free(r->samples);
        r->samples = NULL;
    }
}
