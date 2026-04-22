/*
 * aec_v2.h — 声学回声消除（AEC v2）公共接口与数据结构
 *
 * ═══════════════════════ 算法概述 ═══════════════════════
 *
 * 本模块实现了两级回声消除架构：
 *
 *  ┌──────────────────────────────────────────────────────────┐
 *  │  远端信号 x(n) ──────────────────────────────────────┐   │
 *  │                                                      ↓   │
 *  │  近端信号 d(n) ──► [回声消除 Stage1] ──► e(n) ──► [NLP Stage2] ──► 输出 y(n)
 *  │                         ↑                                │
 *  │                    PBFDAF-NLMS                           │
 *  │                   多分区自适应滤波                        │
 *  └──────────────────────────────────────────────────────────┘
 *
 * Stage 1 — 线性回声消除（EchoSubtraction）
 *   算法：PBFDAF（Partitioned Block Frequency Domain Adaptive Filter）
 *   核心：频域多分区 NLMS 自适应滤波器
 *   滤波器：12 个分区，每分区 64 采样，共覆盖 768 个采样的回声尾
 *   流程：
 *     1. 对远端信号 x 做 FFT，保存到环形 FFT 缓冲（xfBuf，12 分区）
 *     2. 计算回声估计 ŝ = IFFT(Σ X_k · H_k)（12 路加权求和）
 *     3. 计算误差 e = d - ŝ（近端信号减去回声估计）
 *     4. NLMS 更新滤波器系数 H += μ · X* · E / xPow
 *        其中 μ = filter_step_size，误差信号通过 error_threshold 限幅防止发散
 *     5. 如检测到滤波器严重发散（seSum > 20 × sdSum），自动清零重启
 *
 * Stage 2 — 非线性回声抑制（EchoSuppression / NLP）
 *   算法：基于子带相干性的非线性处理
 *   利用三路信号（近端 d、误差 e、远端 x）计算两个相干度量：
 *     cohde = |S_de|² / (S_dd · S_ee)  —— 近端与误差的相干度（越大越像双讲）
 *     cohxd = |S_xd|² / (S_xx · S_dd)  —— 远端与近端的相干度（越大越像纯回声）
 *   状态机：
 *     ● 只有近端说话：cohde≈1，cohxd≈0  → hNl = cohde（基本不压制）
 *     ● 双讲          ：cohde 中等，cohxd 中等 → hNl = min(cohde, 1-cohxd)
 *     ● 纯回声        ：cohde≈0，cohxd≈1  → hNl → 0（大力压制）
 *   抑制增益 hNl 经过 overdrive 非线性放大后乘以误差信号频谱，
 *   再补充舒适噪声，最终经 IFFT + 重叠相加还原时域输出。
 *
 * 参考来源：WebRTC AEC（aec_core.c），在此基础上有定制改动。
 */

#ifndef __AEC_V2_H__
#define __AEC_V2_H__

#include <stdbool.h>
#include <stdint.h>

/* ── 基本尺寸常量 ────────────────────────────────────────── */
#define PART_LEN  64              /* 单分区采样数（时域帧长）               */
#define PART_LEN1 (PART_LEN + 1) /* FFT 唯一系数数（实数 FFT 输出 N/2+1）  */
#define PART_LEN2 (PART_LEN * 2) /* 重叠保存帧长（overlap-save 缓冲长度）  */

/* ── 饱和/限幅宏 ─────────────────────────────────────────── */
#define SPL_WORD16_MAX  32767
#define SPL_WORD16_MIN -32768
#define SPL_MAX(A, B)  (A > B ? A : B)
#define SPL_MIN(A, B)  (A < B ? A : B)
#define SPL_SAT(a, b, c) (b > a ? a : b < c ? c : b)

/* ── 滤波器分区数 ────────────────────────────────────────── */
#define kNormalNumPartitions 12   /* 12 个分区，覆盖 12×64 = 768 采样的回声尾 */

/* ── 外部全局变量 ────────────────────────────────────────── */
extern bool use_sse2_;                              /* 运行时 SSE2 特性标志        */
extern const float kNormalSmoothingCoefficients[2][2]; /* 能量谱平滑系数 [mult-1][0/1] */
extern const float kMinFarendPSD;                   /* 远端 PSD 下界，防止除零     */

/* ── 复数类型 ────────────────────────────────────────────── */
typedef float complex_t[2];  /* [0]=实部，[1]=虚部 */

/* ── 相干性状态（用于 NLP 的功率谱与互功率谱） ────────────── */
typedef struct CoherenceState {
    complex_t sde[PART_LEN1]; /* 近端(d)与误差(e)的互功率谱，用于计算 cohde */
    complex_t sxd[PART_LEN1]; /* 远端(x)与近端(d)的互功率谱，用于计算 cohxd */
    float sx[PART_LEN1];      /* 远端功率谱（平滑后）                         */
    float sd[PART_LEN1];      /* 近端功率谱（平滑后）                         */
    float se[PART_LEN1];      /* 误差信号功率谱（平滑后）                     */
} CoherenceState;

/* ── AEC 主状态结构体 ────────────────────────────────────── */
typedef struct {
    /* ---- 上一帧缓存（overlap-save 用） ---- */
    float nearend_old[PART_LEN]; /* 上一帧近端信号，拼接成 128 点后做 FFT */
    float farend_old[PART_LEN];  /* 上一帧远端信号，拼接成 128 点后做 FFT */

    /* ---- 采样率相关参数 ---- */
    int sampFreq;                /* 采样率（Hz），支持 8000/16000/48000     */
    int mult;                    /* 采样率倍数 = sampFreq/8000，用于时序缩放 */

    /* ---- NLMS 自适应滤波参数 ---- */
    float filter_step_size;      /* NLMS 步长 μ，默认 0.5；8kHz 时 0.6     */
    float error_threshold;       /* 误差限幅阈值，防止滤波器系数溢出         */

    /* ---- 功率谱估计 ---- */
    float xPow[PART_LEN1];       /* 远端信号平滑功率谱（NLMS 归一化分母）    */
    float dPow[PART_LEN1];       /* 近端信号平滑功率谱                       */
    float dMinPow[PART_LEN1];    /* 近端功率谱的追踪最小值（舒适噪声基础）   */
    float dInitMinPow[PART_LEN1];/* 初始阶段渐进的噪声功率谱                 */
    float *noisePow;             /* 当前噪声功率谱指针（指向上面两个之一）   */
    int noiseEstCtr;             /* 噪声估计块计数，前 500*mult 块用初始值   */

    /* ---- 滤波器参数 ---- */
    int num_partitions;          /* 分区数，固定为 kNormalNumPartitions=12  */
    int extreme_filter_divergence; /* 1 = 滤波器严重发散，需清零重启        */
    int xfBufBlockPos;           /* 远端 FFT 环形缓冲的当前写入位置          */

    /* ---- 相干性状态（NLP 用） ---- */
    CoherenceState coherence_state;

    /* ---- 频域缓冲 ---- */
    float xfBuf[2][kNormalNumPartitions * PART_LEN1];  /* 远端 FFT 环形缓冲（实/虚，12×65）  */
    float wfBuf[2][kNormalNumPartitions * PART_LEN1];  /* 自适应滤波器系数频域表示（实/虚）  */
    complex_t xfwBuf[kNormalNumPartitions * PART_LEN1];/* 远端加窗 FFT 缓冲（NLP 用，12×65）*/

    float far_spectrum_buf[kNormalNumPartitions * PART_LEN1]; /* 远端多分区功率谱历史（备用）*/

    /* ---- 时域缓冲 ---- */
    float eBuf[PART_LEN2];       /* 误差信号缓冲（128点，overlap-save）      */
    float outBuf[PART_LEN];      /* 上一帧 NLP 输出后半段（overlap-add 用）  */

    /* ---- 延迟估计 ---- */
    int delayEstCtr;             /* 延迟估计计数器，每 10*mult 块更新一次    */
    int delayIdx;                /* 当前估计的回声延迟（分区索引）           */

    /* ---- 双讲/回声状态检测 ---- */
    short divergeState;          /* 1 = 滤波器发散，直接输出近端信号         */
    float hNlXdAvgMin;           /* cohxd 均值的历史最小值（趋近 0 = 纯回声）*/
    short stNearState;           /* 1 = 检测到只有近端在说话                 */
    short echoState;             /* 1 = 检测到处于回声消除状态               */

    /* ---- NLP overdrive 参数 ---- */
    float overDrive;             /* 目标超驱动系数（由 hNlFbMin 动态计算）   */
    float overdrive_scaling;     /* 平滑后的超驱动系数（缓慢跟踪 overDrive） */
    float hNlFbMin;              /* hNlFbLow 的历史最小值                    */
    float hNlFbLocalMin;         /* hNlFbLow 的局部最小值（缓慢爬升复位）    */
    int hNlNewMin;               /* 1 = 出现了新的更小的 hNlFbLow            */
    int hNlMinCtr;               /* hNlNewMin 出现后的计数（到 2 时更新 overDrive）*/

    /* ---- NLP 模式与舒适噪声 ---- */
    int nlp_mode;                /* NLP 模式 0/1/2，越大压制越强             */
    uint32_t seed;               /* 舒适噪声随机数种子                       */
} AECV2;

/* ═══════════════════ 公共 API ═══════════════════════════ */

/* 创建并初始化 AEC 实例（内部状态清零，默认 16kHz，nlp_mode=1）*/
AECV2 *AECV2_Create(void);

/* 释放 AEC 实例 */
void AECV2_Free(AECV2 *aecv2);

/*
 * 设置初始回声延迟（分区索引，0 ~ num_partitions-1）
 * 若已知硬件延迟可预先设置，加速收敛
 */
void AECV2_Delay_Set(AECV2 *aecv2, int delay);

/*
 * 配置采样率和 NLP 模式（必须在 Create 之后、Process 之前调用）
 *   samplingFreq : 采样率（8000 / 16000 / 48000）
 *   nlpMode      : 0=轻度压制，1=标准，2=强力压制（双讲性能依次变差）
 */
void AECV2_Init(AECV2 *aecv2, int samplingFreq, int nlpMode);

/*
 * 处理一帧音频（每次处理 PART_LEN=64 个采样）
 *   farEnd  : 远端参考信号（播放信号/回采信号），float[-1,1]，64 点
 *   nearEnd : 近端麦克风信号（含回声），float[-1,1]，64 点
 *   output  : 回声消除后的输出，float，64 点
 */
void AECV2_Process(AECV2 *aecv2, float *farEnd, float *nearEnd, float *output);

/* ═══════════════════ 平台加速声明（内部用）══════════════ */

#if defined(ARCH_X86_FAMILY)
void FilterFar_SSE2(int num_partitions, int x_fft_buf_block_pos,
                    float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                    float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                    float y_fft[2][PART_LEN1]);
void ScaleErrorSignal_SSE2(float mu, float error_threshold,
                           float x_pow[PART_LEN1], float ef[2][PART_LEN1]);
void FilterAdaptation_SSE2(int num_partitions, int x_fft_buf_block_pos,
                           float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                           float e_fft[2][PART_LEN1],
                           float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);
int  PartitionDelay_SSE2(int num_partitions,
                         float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);
void UpdateCoherenceSpectra_SSE2(int mult, float efw[2][PART_LEN1],
                                 float dfw[2][PART_LEN1], float xfw[2][PART_LEN1],
                                 CoherenceState *coherence_state,
                                 short *filter_divergence_state,
                                 int *extreme_filter_divergence);
void ComputeCoherence_SSE2(const CoherenceState *coherence_state,
                           float *cohde, float *cohxd);
void Overdrive_SSE2(float overdrive_scaling, float hNlFb, float hNl[PART_LEN1]);
void Suppress_SSE2(const float hNl[PART_LEN1], float efw[2][PART_LEN1]);
#endif

#if defined(MIPS_FPU_LE)
void FilterFar_MIPS(int num_partitions, int x_fft_buf_block_pos,
                    float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                    float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                    float y_fft[2][PART_LEN1]);
void ScaleErrorSignal_MIPS(float mu, float error_threshold,
                           float x_pow[PART_LEN1], float ef[2][PART_LEN1]);
void FilterAdaptation_MIPS(int num_partitions, int x_fft_buf_block_pos,
                           float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                           float e_fft[2][PART_LEN1],
                           float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);
void Overdrive_MIPS(float overdrive_scaling, float hNlFb, float hNl[PART_LEN1]);
void Suppress_MIPS(const float hNl[PART_LEN1], float efw[2][PART_LEN1]);
#endif

#if defined(ARCH_ARM_NEON)
void FilterFar_NEON(int num_partitions, int x_fft_buf_block_pos,
                    float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                    float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                    float y_fft[2][PART_LEN1]);
void ScaleErrorSignal_NEON(float mu, float error_threshold,
                           float x_pow[PART_LEN1], float ef[2][PART_LEN1]);
void FilterAdaptation_NEON(int num_partitions, int x_fft_buf_block_pos,
                           float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                           float e_fft[2][PART_LEN1],
                           float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);
int  PartitionDelay_NEON(int num_partitions,
                         float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);
void UpdateCoherenceSpectra_NEON(int mult, float efw[2][PART_LEN1],
                                 float dfw[2][PART_LEN1], float xfw[2][PART_LEN1],
                                 CoherenceState *coherence_state,
                                 short *filter_divergence_state,
                                 int *extreme_filter_divergence);
void ComputeCoherence_NEON(const CoherenceState *coherence_state,
                           float *cohde, float *cohxd);
void Overdrive_NEON(float overdrive_scaling, float hNlFb, float hNl[PART_LEN1]);
void Suppress_NEON(const float hNl[PART_LEN1], float efw[2][PART_LEN1]);
#endif

/* ── 内存对齐宏（MSVC 与 GCC 兼容） ─────────────────────── */
#ifdef _MSC_VER
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#else
#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))
#endif

/* 查找表（定义在 aec_v2.c，供 NLP overdrive 使用） */
extern ALIGN16_BEG const float ALIGN16_END weightCurve[65];   /* NLP 子带加权系数，高频权重更大  */
extern ALIGN16_BEG const float ALIGN16_END overDriveCurve[65];/* Overdrive 子带指数曲线，高频衰减更强 */

#endif /* __AEC_V2_H__ */
