/*
 * main_realtime.c — 实时 AEC 处理（PortAudio）
 *
 * 通道布局（默认，对应你的设备）：
 *   ch 0-3 : 原始麦克风（近端，near-end）
 *   ch 4   : 级联通道
 *   ch 5   : 回采通道（远端参考，far-end reference）
 *   ch 6   : DSP 处理后通道
 *
 * 用法：
 *   aec_rt                                     列出所有音频设备
 *   aec_rt <input_dev> [near=0] [ref=5] [total_ch=7] [output_dev=-1]
 *
 *     input_dev  : 输入设备索引（从列表中选择你的多通道设备）
 *     near_ch    : 近端麦克风通道号（0-indexed，默认 0）
 *     ref_ch     : 回采/远端参考通道号（0-indexed，默认 5）
 *     total_ch   : 输入设备总通道数（默认 7）
 *     output_dev : 输出设备索引（-1=系统默认，-2=不播放）
 *
 * 处理流程：
 *   Pa_ReadStream(7ch) → 解交织提取 near[64] / ref[64]
 *   → AECV2_Process(ref, near, out)
 *   → Pa_WriteStream(mono out[64])
 *   → 终端实时电平表
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>

#include "portaudio.h"
#include "aec_v2.h"

/* ── 全局退出标志（Ctrl+C 触发） ─────────────────────── */
static volatile int g_running = 1;
static void on_sigint(int s) { (void)s; g_running = 0; }

/* ================================================================
 * 设备信息
 * ================================================================ */
static void list_devices(void) {
    int n = Pa_GetDeviceCount();
    if (n < 0) { fprintf(stderr, "Pa_GetDeviceCount: %s\n", Pa_GetErrorText(n)); return; }

    int default_in  = Pa_GetDefaultInputDevice();
    int default_out = Pa_GetDefaultOutputDevice();

    printf("Audio devices (%d total):\n", n);
    printf("  %-4s  %-50s  %4s  %4s  %s\n", "Idx", "Name", "In", "Out", "Rate");
    printf("  %s\n", "-------------------------------------------------------------------------------------");
    for (int i = 0; i < n; i++) {
        const PaDeviceInfo *d = Pa_GetDeviceInfo(i);
        const char *tag = (i == default_in) ? "<IN> " : (i == default_out) ? "<OUT>" : "     ";
        printf("  [%2d] %s %-46s  %4d  %4d  %.0f Hz\n",
               i, tag, d->name,
               d->maxInputChannels, d->maxOutputChannels,
               d->defaultSampleRate);
    }
    printf("\n");
}

/* ================================================================
 * 音频工具
 * ================================================================ */
static float rms_value(const float *buf, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)buf[i] * buf[i];
    return (float)sqrt(s / n + 1e-12);
}

static float to_dbfs(float r) {
    return 20.0f * log10f(r);
}

#define METER_WIDTH 40

/* ASCII 电平条（范围 -60 ~ 0 dBFS） */
static void print_meter(const char *label, float db) {
    int bars = (int)((db + 60.0f) * METER_WIDTH / 60.0f);
    if (bars < 0)             bars = 0;
    if (bars > METER_WIDTH)   bars = METER_WIDTH;

    char bar[METER_WIDTH + 1];
    for (int i = 0; i < METER_WIDTH; i++) bar[i] = (i < bars) ? '#' : ' ';
    bar[METER_WIDTH] = '\0';

    printf("  %-10s [%s] %+6.1f dBFS\n", label, bar, db);
}

/* ================================================================
 * 主函数
 * ================================================================ */
int main(int argc, char *argv[]) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "Pa_Initialize failed: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    /* ── 无参数：列出设备并退出 ── */
    if (argc < 2) {
        list_devices();
        printf("Usage: %s <input_dev> [near_ch=0] [ref_ch=5] [total_ch=7] [output_dev=-1]\n\n", argv[0]);
        printf("Example (device 3, default channels):\n");
        printf("  %s 3\n\n", argv[0]);
        printf("Example (custom channels, no playback):\n");
        printf("  %s 3 0 5 7 -2\n\n", argv[0]);
        Pa_Terminate();
        return 0;
    }

    /* ── 解析参数 ── */
    int in_dev   = atoi(argv[1]);
    int near_ch  = (argc > 2) ? atoi(argv[2]) : 0;
    int ref_ch   = (argc > 3) ? atoi(argv[3]) : 5;
    int total_ch = (argc > 4) ? atoi(argv[4]) : 7;
    int out_dev_arg = (argc > 5) ? atoi(argv[5]) : -1;

    int do_playback = (out_dev_arg >= -1);
    int out_dev = (out_dev_arg == -1) ? (int)Pa_GetDefaultOutputDevice() : out_dev_arg;

    /* ── 参数校验 ── */
    if (in_dev < 0 || in_dev >= Pa_GetDeviceCount()) {
        fprintf(stderr, "Invalid input device index: %d\n", in_dev);
        Pa_Terminate(); return 1;
    }
    if (near_ch < 0 || near_ch >= total_ch || ref_ch < 0 || ref_ch >= total_ch) {
        fprintf(stderr, "Channel out of range: near_ch=%d ref_ch=%d total_ch=%d\n",
                near_ch, ref_ch, total_ch);
        Pa_Terminate(); return 1;
    }

    const PaDeviceInfo *in_info = Pa_GetDeviceInfo(in_dev);
    if (in_info->maxInputChannels < total_ch) {
        fprintf(stderr, "Device only has %d input channels, but total_ch=%d\n",
                in_info->maxInputChannels, total_ch);
        Pa_Terminate(); return 1;
    }

    /* ── 打印配置摘要 ── */
    printf("=== Real-time AEC ===\n");
    printf("Input  : [%d] %s\n", in_dev, in_info->name);
    printf("  total_ch=%d  near_ch=%d  ref_ch=%d\n", total_ch, near_ch, ref_ch);
    printf("Sample : 48000 Hz, %d samples/block (%.1f ms/block)\n",
           PART_LEN, (float)PART_LEN / 48000.0f * 1000.0f);
    printf("AEC    : %d partitions, filter tail=%.1f ms\n",
           kNormalNumPartitions, (float)(kNormalNumPartitions * PART_LEN) / 48000.0f * 1000.0f);

    /* ── 打开输入流（多通道，阻塞模式） ── */
    PaStreamParameters inp;
    memset(&inp, 0, sizeof(inp));
    inp.device                    = in_dev;
    inp.channelCount              = total_ch;
    inp.sampleFormat              = paFloat32;
    inp.suggestedLatency          = in_info->defaultLowInputLatency;
    inp.hostApiSpecificStreamInfo = NULL;

    PaStream *in_stream = NULL;
    err = Pa_OpenStream(&in_stream, &inp, NULL,
                        48000.0, PART_LEN, paNoFlag, NULL, NULL);
    if (err != paNoError) {
        fprintf(stderr, "Open input stream failed: %s\n", Pa_GetErrorText(err));
        fprintf(stderr, "Tips: try WASAPI exclusive mode or check device channel count.\n");
        Pa_Terminate(); return 1;
    }

    /* ── 打开输出流（单声道，阻塞模式） ── */
    PaStream *out_stream = NULL;
    if (do_playback && out_dev >= 0 && out_dev < Pa_GetDeviceCount()) {
        const PaDeviceInfo *out_info = Pa_GetDeviceInfo(out_dev);
        printf("Output : [%d] %s\n", out_dev, out_info->name);

        PaStreamParameters outp;
        memset(&outp, 0, sizeof(outp));
        outp.device                    = out_dev;
        outp.channelCount              = 1;
        outp.sampleFormat              = paFloat32;
        outp.suggestedLatency          = out_info->defaultLowOutputLatency;
        outp.hostApiSpecificStreamInfo = NULL;

        err = Pa_OpenStream(&out_stream, NULL, &outp,
                            48000.0, PART_LEN, paNoFlag, NULL, NULL);
        if (err != paNoError) {
            fprintf(stderr, "Open output stream failed (%s), running without playback.\n",
                    Pa_GetErrorText(err));
            out_stream = NULL;
        }
    } else {
        printf("Output : disabled\n");
    }
    printf("\n");

    /* ── AEC 初始化 ── */
    AECV2 *aec = AECV2_Create();
    AECV2_Init(aec, 48000, 1);

    /* ── 启动流 ── */
    Pa_StartStream(in_stream);
    if (out_stream) Pa_StartStream(out_stream);

    signal(SIGINT, on_sigint);

    /* ── 缓冲区分配 ── */
    /* 输入：total_ch 通道交织，每通道 PART_LEN 个采样 */
    float *in_buf  = (float*)malloc(sizeof(float) * PART_LEN * total_ch);
    float  near_buf[PART_LEN];
    float  ref_buf[PART_LEN];
    float  out_buf[PART_LEN];

    printf("Processing... (Ctrl+C to stop)\n\n\n\n"); /* 预留 4 行给电平表 */

    long frame_cnt  = 0;
    long overflow_cnt = 0;

    while (g_running) {
        /* ① 读取一帧多通道音频（阻塞直到数据就绪） */
        err = Pa_ReadStream(in_stream, in_buf, PART_LEN);
        if (err == paInputOverflowed) {
            overflow_cnt++;
            /* 缓冲区溢出：继续处理，数据可能有跳变 */
        } else if (err != paNoError) {
            fprintf(stderr, "\nPa_ReadStream error: %s\n", Pa_GetErrorText(err));
            break;
        }

        /* ② 解交织：从交织流中提取目标通道
         *    输入格式：[ch0_s0, ch1_s0, ..., chN_s0,  ch0_s1, ch1_s1, ..., chN_s1, ...]
         */
        for (int i = 0; i < PART_LEN; i++) {
            near_buf[i] = in_buf[i * total_ch + near_ch];
            ref_buf[i]  = in_buf[i * total_ch + ref_ch];
        }

        /* ③ AEC 处理
         *    注意参数顺序：AECV2_Process(aec, farEnd, nearEnd, output)
         *    far-end = 回采（ref），near-end = 麦克风（near）
         */
        AECV2_Process(aec, ref_buf, near_buf, out_buf);

        /* ④ 播放输出 */
        if (out_stream) {
            err = Pa_WriteStream(out_stream, out_buf, PART_LEN);
            if (err == paOutputUnderflowed) {
                /* 欠载：继续运行 */
            } else if (err != paNoError) {
                fprintf(stderr, "\nPa_WriteStream error: %s\n", Pa_GetErrorText(err));
                break;
            }
        }

        /* ⑤ 每 150 帧刷新一次电平表（约 200ms @ 48kHz/64） */
        frame_cnt++;
        if (frame_cnt % 150 == 0) {
            float db_near = to_dbfs(rms_value(near_buf, PART_LEN));
            float db_ref  = to_dbfs(rms_value(ref_buf,  PART_LEN));
            float db_out  = to_dbfs(rms_value(out_buf,  PART_LEN));

            /* ANSI 转义：上移 4 行覆盖，Windows Terminal / PowerShell 支持 */
            printf("\033[4A");
            printf("  Frames: %-8ld  Overflows: %-4ld\n", frame_cnt, overflow_cnt);
            print_meter("Near-end",  db_near);
            print_meter("Far(ref)",  db_ref);
            print_meter("AEC out",   db_out);
            fflush(stdout);
        }
    }

    /* ── 清理 ── */
    printf("\nStopping...\n");
    Pa_StopStream(in_stream);
    Pa_CloseStream(in_stream);
    if (out_stream) {
        Pa_StopStream(out_stream);
        Pa_CloseStream(out_stream);
    }
    AECV2_Free(aec);
    free(in_buf);
    Pa_Terminate();

    printf("Total frames processed: %ld (%.1f seconds)\n",
           frame_cnt, (float)frame_cnt * PART_LEN / 48000.0f);
    return 0;
}
