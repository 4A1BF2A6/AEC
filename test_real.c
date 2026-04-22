/*
 * aec_wav.exe - Process real multi-channel WAV through AEC
 *
 * Channel layout (default for this device):
 *   ch 0-3 : raw mic (near-end)
 *   ch 4   : cascaded
 *   ch 5   : loopback / far-end reference (回采)
 *   ch 6   : DSP-processed output (reference for comparison)
 *
 * Usage:
 *   aec_wav <input.wav> [near_ch=0] [ref_ch=5] [dsp_ch=6] [output=output.wav]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "aec_v2.h"

/* ------------------------------------------------------------------ */
/* WAV I/O                                                              */
/* ------------------------------------------------------------------ */

#define WAV_FMT_PCM   1
#define WAV_FMT_FLOAT 3
#define WAV_FMT_EXT   0xFFFE

typedef struct {
    int    sample_rate;
    int    num_channels;
    int    bits_per_sample;
    int    audio_format;    /* WAV_FMT_PCM / WAV_FMT_FLOAT */
    int    num_samples;     /* per channel */
    float **data;           /* data[ch][sample], normalized to [-1, 1] */
} WavData;

static uint16_t read_le16(FILE *f) {
    uint8_t b[2]; fread(b, 1, 2, f);
    return (uint16_t)(b[0] | (b[1] << 8));
}
static uint32_t read_le32(FILE *f) {
    uint8_t b[4]; fread(b, 1, 4, f);
    return (uint32_t)(b[0] | (b[1]<<8) | (b[2]<<16) | (b[3]<<24));
}
static void write_le16(FILE *f, uint16_t v) {
    uint8_t b[2] = {v & 0xFF, (v >> 8) & 0xFF}; fwrite(b, 1, 2, f);
}
static void write_le32(FILE *f, uint32_t v) {
    uint8_t b[4] = {v&0xFF,(v>>8)&0xFF,(v>>16)&0xFF,(v>>24)&0xFF}; fwrite(b, 1, 4, f);
}

WavData *wav_read(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }

    char tag[4];
    /* RIFF header */
    if (fread(tag, 1, 4, f) != 4 || memcmp(tag, "RIFF", 4)) { fclose(f); return NULL; }
    read_le32(f); /* file size */
    if (fread(tag, 1, 4, f) != 4 || memcmp(tag, "WAVE", 4)) { fclose(f); return NULL; }

    WavData *wav = (WavData*)calloc(1, sizeof(WavData));
    int fmt_ok = 0;
    uint32_t data_off = 0, data_sz = 0;

    /* Scan chunks */
    while (fread(tag, 1, 4, f) == 4) {
        uint32_t chunk_size = read_le32(f);
        long chunk_start = ftell(f);

        if (memcmp(tag, "fmt ", 4) == 0) {
            uint16_t audio_fmt = read_le16(f);
            uint16_t nch       = read_le16(f);
            uint32_t sr        = read_le32(f);
            read_le32(f); /* byte rate */
            read_le16(f); /* block align */
            uint16_t bps       = read_le16(f);

            wav->audio_format   = audio_fmt;
            wav->num_channels   = nch;
            wav->sample_rate    = sr;
            wav->bits_per_sample = bps;

            /* WAVE_FORMAT_EXTENSIBLE: read SubFormat */
            if (audio_fmt == WAV_FMT_EXT && chunk_size >= 40) {
                read_le16(f); /* cbSize */
                read_le16(f); /* wValidBitsPerSample */
                read_le32(f); /* dwChannelMask */
                uint16_t subfmt = read_le16(f); /* first 2 bytes of SubFormat GUID */
                wav->audio_format = subfmt; /* PCM=1 or FLOAT=3 */
            }
            fmt_ok = 1;

        } else if (memcmp(tag, "data", 4) == 0) {
            data_off = (uint32_t)ftell(f);
            data_sz  = chunk_size;
            break; /* data is always last meaningful chunk */
        }

        fseek(f, chunk_start + (long)chunk_size, SEEK_SET);
    }

    if (!fmt_ok || !data_off) {
        fprintf(stderr, "Invalid or unsupported WAV file\n");
        free(wav); fclose(f); return NULL;
    }

    int bytes_per_sample = wav->bits_per_sample / 8;
    wav->num_samples = (int)(data_sz / ((uint32_t)bytes_per_sample * wav->num_channels));

    /* Allocate per-channel arrays */
    wav->data = (float**)malloc(wav->num_channels * sizeof(float*));
    for (int c = 0; c < wav->num_channels; c++)
        wav->data[c] = (float*)malloc(wav->num_samples * sizeof(float));

    fseek(f, data_off, SEEK_SET);

    /* Read interleaved samples and deinterleave */
    for (int s = 0; s < wav->num_samples; s++) {
        for (int c = 0; c < wav->num_channels; c++) {
            float val = 0.0f;
            if (wav->bits_per_sample == 16 && wav->audio_format == WAV_FMT_PCM) {
                int16_t v; fread(&v, 2, 1, f);
                val = v / 32768.0f;
            } else if (wav->bits_per_sample == 24 && wav->audio_format == WAV_FMT_PCM) {
                uint8_t b[3]; fread(b, 1, 3, f);
                int32_t v = (int32_t)(((uint32_t)b[2]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[0]<<8));
                val = (v >> 8) / 8388608.0f;
            } else if (wav->bits_per_sample == 32 && wav->audio_format == WAV_FMT_PCM) {
                int32_t v; fread(&v, 4, 1, f);
                val = v / 2147483648.0f;
            } else if (wav->bits_per_sample == 32 && wav->audio_format == WAV_FMT_FLOAT) {
                fread(&val, 4, 1, f);
            } else {
                /* Skip unsupported format bytes */
                fseek(f, bytes_per_sample, SEEK_CUR);
            }
            wav->data[c][s] = val;
        }
    }

    fclose(f);
    return wav;
}

void wav_free(WavData *wav) {
    if (!wav) return;
    for (int c = 0; c < wav->num_channels; c++) free(wav->data[c]);
    free(wav->data); free(wav);
}

int wav_write_mono(const char *path, const float *data, int n, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write: %s\n", path); return 0; }
    uint32_t data_sz = (uint32_t)n * 2;
    fwrite("RIFF", 1, 4, f); write_le32(f, 36 + data_sz);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); write_le32(f, 16);
    write_le16(f, 1); write_le16(f, 1);            /* PCM, mono */
    write_le32(f, (uint32_t)sr);
    write_le32(f, (uint32_t)sr * 2);               /* byte rate */
    write_le16(f, 2); write_le16(f, 16);           /* block align, bps */
    fwrite("data", 1, 4, f); write_le32(f, data_sz);
    for (int i = 0; i < n; i++) {
        float s = data[i];
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t v = (int16_t)(s * 32767.0f);
        fwrite(&v, 2, 1, f);
    }
    fclose(f); return 1;
}

/* ------------------------------------------------------------------ */
/* Metrics                                                              */
/* ------------------------------------------------------------------ */

static double rms_db(const float *x, int n) {
    double p = 0;
    for (int i = 0; i < n; i++) p += (double)x[i] * x[i];
    return 20.0 * log10(sqrt(p / n) + 1e-12);
}

static double reduction_db(const float *in, const float *out, int n) {
    double pi = 0, po = 0;
    for (int i = 0; i < n; i++) {
        pi += (double)in[i]  * in[i];
        po += (double)out[i] * out[i];
    }
    return 10.0 * log10((po + 1e-12) / (pi + 1e-12));
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input.wav> [near_ch=0] [ref_ch=5] [dsp_ch=6] [output.wav=output.wav]\n", argv[0]);
        printf("\nDefault channel layout:\n");
        printf("  ch 0-3 : raw mic (near-end)\n");
        printf("  ch 4   : cascaded\n");
        printf("  ch 5   : loopback / far-end reference (回采)\n");
        printf("  ch 6   : DSP hardware processed (comparison reference)\n");
        printf("\nPass -1 for dsp_ch to skip DSP comparison.\n");
        return 1;
    }

    const char *input  = argv[1];
    int near_ch = (argc > 2) ? atoi(argv[2]) : 0;
    int ref_ch  = (argc > 3) ? atoi(argv[3]) : 5;
    int dsp_ch  = (argc > 4) ? atoi(argv[4]) : 6;
    const char *output = (argc > 5) ? argv[5] : "output.wav";

    /* Read input */
    printf("Reading: %s\n", input);
    WavData *wav = wav_read(input);
    if (!wav) return 1;

    printf("Format : %d Hz, %d channels, %d-bit %s\n",
           wav->sample_rate, wav->num_channels, wav->bits_per_sample,
           wav->audio_format == WAV_FMT_FLOAT ? "float" : "PCM");
    printf("Duration: %.2f s  (%d samples/ch)\n",
           (double)wav->num_samples / wav->sample_rate, wav->num_samples);

    /* Validate channels */
    if (near_ch < 0 || near_ch >= wav->num_channels) {
        fprintf(stderr, "near_ch=%d out of range (file has %d channels)\n", near_ch, wav->num_channels);
        wav_free(wav); return 1;
    }
    if (ref_ch < 0 || ref_ch >= wav->num_channels) {
        fprintf(stderr, "ref_ch=%d out of range (file has %d channels)\n", ref_ch, wav->num_channels);
        wav_free(wav); return 1;
    }

    printf("Near-end ch : %d\n", near_ch);
    printf("Far-end ref : %d\n", ref_ch);
    if (dsp_ch >= 0 && dsp_ch < wav->num_channels)
        printf("DSP ref ch  : %d\n", dsp_ch);

    /* Initialize AEC */
    AECV2 *aec = AECV2_Create();
    AECV2_Init(aec, wav->sample_rate, 1);
    printf("\nAEC: %d Hz, %d partitions, mult=%d, filter tail=%.1f ms\n",
           wav->sample_rate, aec->num_partitions, aec->mult,
           (double)(aec->num_partitions * PART_LEN) / wav->sample_rate * 1000.0);

    /* Process block by block */
    int total_samples = wav->num_samples;
    int blocks = total_samples / PART_LEN;
    float *out_buf = (float*)calloc(total_samples, sizeof(float));

    float far_block[PART_LEN], near_block[PART_LEN], out_block[PART_LEN];

    printf("\nProcessing %d blocks (%d samples each)...\n", blocks, PART_LEN);
    for (int b = 0; b < blocks; b++) {
        int off = b * PART_LEN;
        memcpy(far_block,  wav->data[ref_ch]  + off, PART_LEN * sizeof(float));
        memcpy(near_block, wav->data[near_ch] + off, PART_LEN * sizeof(float));
        AECV2_Process(aec, far_block, near_block, out_block);
        memcpy(out_buf + off, out_block, PART_LEN * sizeof(float));
    }
    printf("Done.\n\n");

    /* Write outputs */
    wav_write_mono(output, out_buf, total_samples, wav->sample_rate);
    printf("AEC output  : %s\n", output);

    /* Also extract input channels for listening comparison */
    char near_wav[256], ref_wav[256];
    snprintf(near_wav, sizeof(near_wav), "near_ch%d.wav", near_ch);
    snprintf(ref_wav,  sizeof(ref_wav),  "ref_ch%d.wav",  ref_ch);
    wav_write_mono(near_wav, wav->data[near_ch], total_samples, wav->sample_rate);
    wav_write_mono(ref_wav,  wav->data[ref_ch],  total_samples, wav->sample_rate);
    printf("Near-end    : %s\n", near_wav);
    printf("Far-end ref : %s\n", ref_wav);

    if (dsp_ch >= 0 && dsp_ch < wav->num_channels) {
        char dsp_wav[256];
        snprintf(dsp_wav, sizeof(dsp_wav), "dsp_ch%d.wav", dsp_ch);
        wav_write_mono(dsp_wav, wav->data[dsp_ch], total_samples, wav->sample_rate);
        printf("DSP ref     : %s\n", dsp_wav);
    }

    /* Metrics */
    printf("\n--- Signal Metrics ---\n");
    printf("Near-end RMS   : %6.1f dBFS\n", rms_db(wav->data[near_ch], total_samples));
    printf("Far-end ref RMS: %6.1f dBFS\n", rms_db(wav->data[ref_ch],  total_samples));
    printf("AEC output RMS : %6.1f dBFS\n", rms_db(out_buf, total_samples));
    printf("Echo reduction (AEC vs near-end): %+.1f dB\n",
           reduction_db(wav->data[near_ch], out_buf, total_samples));

    if (dsp_ch >= 0 && dsp_ch < wav->num_channels) {
        printf("DSP output RMS : %6.1f dBFS\n", rms_db(wav->data[dsp_ch], total_samples));
        printf("Echo reduction (DSP vs near-end): %+.1f dB\n",
               reduction_db(wav->data[near_ch], wav->data[dsp_ch], total_samples));
    }

    AECV2_Free(aec);
    wav_free(wav);
    free(out_buf);
    return 0;
}
