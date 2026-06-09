/*
 * aec_v2.c — 声学回声消除核心实现
 *
 * 整体处理流程（每调用一次 AECV2_Process 处理 64 个采样）：
 *
 *  AECV2_Process
 *   ├─ 1. 远端/近端信号拼接上帧构成 128 点，FFT → farend_fft / nearend_fft
 *   ├─ 2. 更新远端功率谱 xPow、近端功率谱 dPow
 *   ├─ 3. 更新噪声估计 dMinPow（最小值追踪），用于舒适噪声
 *   ├─ 4. EchoSubtraction（线性回声消除 Stage1）
 *   │    ├─ 更新远端 FFT 环形缓冲 xfBuf（PBFDAF 用）
 *   │    ├─ FilterFar：ŝ = Σ X_k·H_k（12 分区卷积，频域）
 *   │    ├─ IFFT(ŝ) 得时域回声估计 s，误差 e = d - s
 *   │    ├─ ScaleErrorSignal：NLMS 归一化 + 步长 μ + 误差限幅
 *   │    └─ FilterAdaptation：H += IFFT(X*·E)→FFT（频域 LMS 系数更新）
 *   └─ 5. EchoSuppression（非线性回声抑制 Stage2 / NLP）
 *        ├─ 近端/误差/远端三路加窗 FFT → dfw / efw / xfw
 *        ├─ UpdateCoherenceSpectra：平滑功率谱 sd/se/sx 和互功率谱 sde/sxd
 *        ├─ ComputeCoherence：cohde（近端-误差相干）、cohxd（近端-远端相干）
 *        ├─ FormSuppressionGain：状态机判断说话状态 → 生成 hNl 抑制增益
 *        ├─ Overdrive：对 hNl 做非线性放大（指数曲线），高频额外压制
 *        ├─ Suppress：efw *= hNl（频域乘法）
 *        ├─ ComfortNoise：注入舒适噪声，避免完全静音的人工感
 *        └─ ScaledInverseFft + overlap-add → 时域输出
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../_thirdparty/ooura_fft/ooura_fft.h"
#include "aec_v2.h"

bool use_sse2_; /* 运行时 SSE2 特性标志，由 AECV2_Create 检测 */

#if defined(ARCH_X86_FAMILY)
// List of features in x86.
typedef enum { kSSE2, kSSE3 } CPUFeature;

#if defined(__pic__) && defined(__i386__)
static inline void __cpuid(int cpu_info[4], int info_type) {
  __asm__ volatile(
      "mov %%ebx, %%edi\n"
      "cpuid\n"
      "xchg %%edi, %%ebx\n"
      : "=a"(cpu_info[0]), "=D"(cpu_info[1]), "=c"(cpu_info[2]),
        "=d"(cpu_info[3])
      : "a"(info_type));
}
#else
static inline void __cpuid(int cpu_info[4], int info_type) {
  __asm__ volatile("cpuid\n"
                   : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]),
                     "=d"(cpu_info[3])
                   : "a"(info_type));
}
#endif

// Actual feature detection for x86.
static int GetCPUInfo(CPUFeature feature) {
  int cpu_info[4];
  __cpuid(cpu_info, 1);
  if (feature == kSSE2) {
    return 0 != (cpu_info[3] & 0x04000000);
  }
  if (feature == kSSE3) {
    return 0 != (cpu_info[2] & 0x00000001);
  }
  return 0;
}
#endif

static const uint32_t kMaxSeedUsed = 0x80000000;

static uint32_t IncreaseSeed(uint32_t* seed) {
  seed[0] = (seed[0] * ((int32_t)69069) + 1) & (kMaxSeedUsed - 1);
  return seed[0];
}

int16_t SPL_RandU(uint32_t* seed) {
  return (int16_t)(IncreaseSeed(seed) >> 16);
}

// Creates an array of uniformly distributed variables.
int16_t SPL_RandUArray(int16_t* vector,
                       int16_t vector_length,
                       uint32_t* seed) {
  int i;
  for (i = 0; i < vector_length; i++) {
    vector[i] = SPL_RandU(seed);
  }
  return vector_length;
}

// Matlab code to produce table:
// win = sqrt(hanning(63)); win = [0 ; win(1:32)];
// fprintf(1, '\t%.14f, %.14f, %.14f,\n', win);
ALIGN16_BEG const float ALIGN16_END sqrtHanning[65] = {
    0.00000000000000f, 0.02454122852291f, 0.04906767432742f, 0.07356456359967f,
    0.09801714032956f, 0.12241067519922f, 0.14673047445536f, 0.17096188876030f,
    0.19509032201613f, 0.21910124015687f, 0.24298017990326f, 0.26671275747490f,
    0.29028467725446f, 0.31368174039889f, 0.33688985339222f, 0.35989503653499f,
    0.38268343236509f, 0.40524131400499f, 0.42755509343028f, 0.44961132965461f,
    0.47139673682600f, 0.49289819222978f, 0.51410274419322f, 0.53499761988710f,
    0.55557023301960f, 0.57580819141785f, 0.59569930449243f, 0.61523159058063f,
    0.63439328416365f, 0.65317284295378f, 0.67155895484702f, 0.68954054473707f,
    0.70710678118655f, 0.72424708295147f, 0.74095112535496f, 0.75720884650648f,
    0.77301045336274f, 0.78834642762661f, 0.80320753148064f, 0.81758481315158f,
    0.83146961230255f, 0.84485356524971f, 0.85772861000027f, 0.87008699110871f,
    0.88192126434835f, 0.89322430119552f, 0.90398929312344f, 0.91420975570353f,
    0.92387953251129f, 0.93299279883474f, 0.94154406518302f, 0.94952818059304f,
    0.95694033573221f, 0.96377606579544f, 0.97003125319454f, 0.97570213003853f,
    0.98078528040323f, 0.98527764238894f, 0.98917650996478f, 0.99247953459871f,
    0.99518472667220f, 0.99729045667869f, 0.99879545620517f, 0.99969881869620f,
    1.00000000000000f};

// Matlab code to produce table:
// weightCurve = [0 ; 0.3 * sqrt(linspace(0,1,64))' + 0.1];
// fprintf(1, '\t%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,\n', weightCurve);
#if 1
ALIGN16_BEG const float ALIGN16_END weightCurve[65] = {
    0.0000f, 0.1000f, 0.1378f, 0.1535f, 0.1655f, 0.1756f, 0.1845f, 0.1926f,
    0.2000f, 0.2069f, 0.2134f, 0.2195f, 0.2254f, 0.2309f, 0.2363f, 0.2414f,
    0.2464f, 0.2512f, 0.2558f, 0.2604f, 0.2648f, 0.2690f, 0.2732f, 0.2773f,
    0.2813f, 0.2852f, 0.2890f, 0.2927f, 0.2964f, 0.3000f, 0.3035f, 0.3070f,
    0.3104f, 0.3138f, 0.3171f, 0.3204f, 0.3236f, 0.3268f, 0.3299f, 0.3330f,
    0.3360f, 0.3390f, 0.3420f, 0.3449f, 0.3478f, 0.3507f, 0.3535f, 0.3563f,
    0.3591f, 0.3619f, 0.3646f, 0.3673f, 0.3699f, 0.3726f, 0.3752f, 0.3777f,
    0.3803f, 0.3828f, 0.3854f, 0.3878f, 0.3903f, 0.3928f, 0.3952f, 0.3976f,
    0.4000f};
#endif

#if 0
ALIGN16_BEG const float ALIGN16_END weightCurve[65] = {
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f};
#endif

#if 0 //jay
ALIGN16_BEG const float ALIGN16_END weightCurve[65] = {
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f, 0.1000f,
    0.1000f};
#endif

// Matlab code to produce table:
// overDriveCurve = [sqrt(linspace(0,1,65))' + 1];
// fprintf(1, '\t%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,\n', overDriveCurve);
#if 1
ALIGN16_BEG const float ALIGN16_END overDriveCurve[65] = {
    1.0000f, 1.1250f, 1.1768f, 1.2165f, 1.2500f, 1.2795f, 1.3062f, 1.3307f,
    1.3536f, 1.3750f, 1.3953f, 1.4146f, 1.4330f, 1.4507f, 1.4677f, 1.4841f,
    1.5000f, 1.5154f, 1.5303f, 1.5449f, 1.5590f, 1.5728f, 1.5863f, 1.5995f,
    1.6124f, 1.6250f, 1.6374f, 1.6495f, 1.6614f, 1.6731f, 1.6847f, 1.6960f,
    1.7071f, 1.7181f, 1.7289f, 1.7395f, 1.7500f, 1.7603f, 1.7706f, 1.7806f,
    1.7906f, 1.8004f, 1.8101f, 1.8197f, 1.8292f, 1.8385f, 1.8478f, 1.8570f,
    1.8660f, 1.8750f, 1.8839f, 1.8927f, 1.9014f, 1.9100f, 1.9186f, 1.9270f,
    1.9354f, 1.9437f, 1.9520f, 1.9601f, 1.9682f, 1.9763f, 1.9843f, 1.9922f, 2.0000f};
#endif

#if 0
ALIGN16_BEG const float ALIGN16_END overDriveCurve[65] = {
  2.0000f,
  1.9922f,1.9843f,1.9763f,1.9682f,1.9601f,1.9520f,1.9437f,1.9354f,
  1.9270f,1.9186f,1.9100f,1.9014f,1.8927f,1.8839f,1.8750f,1.8660f,
  1.8570f,1.8478f,1.8385f,1.8292f,1.8197f,1.8101f,1.8004f,1.7906f,
  1.7806f,1.7706f,1.7603f,1.7500f,1.7395f,1.7289f,1.7181f,1.7071f,
  1.6960f,1.6847f,1.6731f,1.6614f,1.6495f,1.6374f,1.6250f,1.6124f,
  1.5995f,1.5863f,1.5728f,1.5590f,1.5449f,1.5303f,1.5154f,1.5000f,
  1.4841f,1.4677f,1.4507f,1.4330f,1.4146f,1.3953f,1.3750f,1.3536f,
  1.3307f,1.3062f,1.2795f,1.2500f,1.2165f,1.1768f,1.1250f,1.0000f
};
#endif

#if 0
ALIGN16_BEG const float ALIGN16_END overDriveCurve[65] = {
  3.0000f,
  3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,
  3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,
  3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,
  3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,
  3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,
  3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,
  3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,
  3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f,3.0000f
};
#endif

#if 0 //jay
ALIGN16_BEG const float ALIGN16_END overDriveCurve[65] = {
    1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
    1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
    1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
    1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
    1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
    1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
    1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
    1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
    1.0000f};
#endif

const float kNormalSmoothingCoefficients[2][2] = {{0.9f, 0.1f}, {0.93f, 0.07f}};

// Threshold to protect against the ill-effects of a zero far-end.
const float kMinFarendPSD = 15;    

// Number of partitions forming the NLP's "preferred" bands.
enum { kPrefBandSize = 24 };

static const float kNormalMinOverDrive[3] = {1.0f, 2.0f, 5.0f};

// Target suppression levels for nlp modes.
// log{0.001, 0.00001, 0.00000001}
static const float kTargetSupp[3] = {-6.9f, -11.5f, -18.4f};

static int CmpFloat(const void* a, const void* b) {
  const float* da = (const float*)a;
  const float* db = (const float*)b;

  return (*da > *db) - (*da < *db);
}

/*
 * 对 128 点时域信号加 sqrt(Hanning) 窗
 * overlap-save 中窗函数分两半：
 *   前 64 点：x[i]          * sqrtHanning[i]         （上升沿）
 *   后 64 点：x[PART_LEN+i] * sqrtHanning[PART_LEN-i]（下降沿）
 * 两帧拼接后加窗，时域乘以窗函数 = 频域卷积的辅助手段，减少 FFT 泄漏
 */
__inline static void WindowData(float* x_windowed, const float* x) {
  int i;
  for (i = 0; i < PART_LEN; i++) {
    x_windowed[i] = x[i] * sqrtHanning[i];
    x_windowed[PART_LEN + i] = x[PART_LEN + i] * sqrtHanning[PART_LEN - i];
  }
}

// Puts fft output data into a complex valued array.
__inline static void StoreAsComplex(const float* data, float data_complex[2][PART_LEN1]) {
  int i;
  data_complex[0][0] = data[0];
  data_complex[1][0] = 0;
  for (i = 1; i < PART_LEN; i++) {
    data_complex[0][i] = data[2 * i];
    data_complex[1][i] = data[2 * i + 1];
  }
  data_complex[0][PART_LEN] = data[1];
  data_complex[1][PART_LEN] = 0;
}

static void Fft(float time_data[PART_LEN2], float freq_data[2][PART_LEN1]) {
  int i;
  OouraFft_Fft(time_data);

  // Reorder fft output data.
  freq_data[1][0] = 0;           //虚部
  freq_data[1][PART_LEN] = 0;    //虚部
  freq_data[0][0] = time_data[0]; //实部
  freq_data[0][PART_LEN] = time_data[1]; //实部
  for (i = 1; i < PART_LEN; i++) {
    freq_data[0][i] = time_data[2 * i];
    freq_data[1][i] = time_data[2 * i + 1];
  }
}

static void ScaledInverseFft(float freq_data[2][PART_LEN1],
                             float time_data[PART_LEN2],
                             float scale,
                             int conjugate) {
  int i;
  const float normalization = scale / (float)(PART_LEN2);  // normalization = 2/128
  const float sign = (conjugate ? -1.0f : 1.0f);
  time_data[0] = freq_data[0][0] * normalization;
  time_data[1] = freq_data[0][PART_LEN] * normalization;
  for (i = 1; i < PART_LEN; i++) {
    time_data[2 * i] = freq_data[0][i] * normalization;
    time_data[2 * i + 1] = sign * freq_data[1][i] * normalization;
  }
  OouraFft_InverseFft(time_data);
}

__inline static float MulRe(float aRe, float aIm, float bRe, float bIm) {
  return aRe * bRe - aIm * bIm;
}

__inline static float MulIm(float aRe, float aIm, float bRe, float bIm) {
  return aRe * bIm + aIm * bRe;
}

/*
 * Stage1 步骤①：频域回声估计  ŝ(f) = Σ_{k=0}^{11} X_k(f) · H_k(f)
 *
 * PBFDAF 核心，将 12 分区的远端 FFT 历史与滤波器系数复数相乘累加。
 * 等价于时域 12×64 点的 FIR 长卷积，但在频域分区完成效率更高。
 *
 * 参数：
 *   x_fft_buf_block_pos : 环形缓冲写指针（最新帧位置）
 *   x_fft_buf           : 远端 FFT 环形缓冲（实/虚，12×65）
 *   h_fft_buf           : 自适应滤波器系数频域表示（12 分区）
 *   y_fft               : 输出回声估计频谱（调用前须清零）
 */
static void FilterFar_C(int num_partitions,
                      int x_fft_buf_block_pos,
                      float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float y_fft[2][PART_LEN1]) {
  int i;
  //x_fft_buf_block_pos代表�?后一组插入的数据index
  for (i = 0; i < num_partitions; i++) {
    int j;
    int xPos = (i + x_fft_buf_block_pos) * PART_LEN1;
    int pos = i * PART_LEN1;
    // Check for wrap
    if (i + x_fft_buf_block_pos >= num_partitions) {
      xPos -= num_partitions * (PART_LEN1);
    }

    //y = h * x
    //复数相乘
    for (j = 0; j < PART_LEN1; j++) {
      y_fft[0][j] += MulRe(x_fft_buf[0][xPos + j], x_fft_buf[1][xPos + j], h_fft_buf[0][pos + j], h_fft_buf[1][pos + j]);
      y_fft[1][j] += MulIm(x_fft_buf[0][xPos + j], x_fft_buf[1][xPos + j], h_fft_buf[0][pos + j], h_fft_buf[1][pos + j]);
    }
  }
}

static void FilterFar(int num_partitions,
                      int x_fft_buf_block_pos,
                      float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float y_fft[2][PART_LEN1]) {
  #if defined(MIPS_FPU_LE)
  FilterFar_MIPS(num_partitions,x_fft_buf_block_pos,x_fft_buf,h_fft_buf,y_fft);
  #elif defined(ARCH_ARM_NEON)
  FilterFar_NEON(num_partitions,x_fft_buf_block_pos,x_fft_buf,h_fft_buf,y_fft);
  #elif defined(ARCH_X86_FAMILY)
  if (use_sse2_) {
    FilterFar_SSE2(num_partitions,x_fft_buf_block_pos,x_fft_buf,h_fft_buf,y_fft);
  } 
  else {
    FilterFar_C(num_partitions,x_fft_buf_block_pos,x_fft_buf,h_fft_buf,y_fft);
  }
  #else
  FilterFar_C(num_partitions,x_fft_buf_block_pos,x_fft_buf,h_fft_buf,y_fft);
  #endif
}


/*
 * Stage1 步骤②：NLMS 误差信号缩放（归一化 + 步长 + 限幅）
 *
 * 标准 NLMS 更新公式：ΔH = μ · E(f) / xPow(f)
 * 本函数对误差频谱 ef 做原地修改：ef = μ · clip(ef / xPow)
 *
 *   ① ef /= xPow：归一化，消除远端信号幅度对步长的影响
 *   ② 若 |ef| > error_threshold：将 ef 缩放至阈值附近
 *      作用一：防止滤波器系数溢出（远端近零时 xPow≈0 导致 ef 暴涨）
 *      作用二：双讲时误差突然变大，限幅防止步长过大破坏已收敛的系数
 *   ③ ef *= μ：乘以步长（默认 0.5，越大收敛越快但越不稳定）
 *
 * 参数：
 *   mu              : NLMS 步长，默认 0.5
 *   error_threshold : 误差限幅阈值，默认 1.5e-6
 *   x_pow           : 远端功率谱（已平滑），作为 NLMS 归一化分母
 *   ef              : 误差信号频谱（原地修改，输出为缩放后的 ΔH 频域）
 */
static void ScaleErrorSignal_C(float mu,
                             float error_threshold,
                             float x_pow[PART_LEN1],
                             float ef[2][PART_LEN1]) {
  // NLMS可变步长
  // filter_step_size 0.5
  // error_threshold 1.5e-6f
  // xPow[i] = 0.9*xPow[i] + 0.1*12*far_spectrum 远端信号能量相关
  // ef误差信号频域
  // �?组数�?:
  // 1.ef[0][i] ef[1][i] x_pow[i] 87.560715 584.836548 14325.222656
  // 2.ef[0][i] ef[1][i] abs_ef   0.006112 0.040826 0.041281
  // 3.abs_ef 0.000036
  // 4.ef[0][i] ef[1][i] 0.000000 0.000001
  int i;
  float abs_ef;
  for (i = 0; i < (PART_LEN1); i++) {
    //printf("1.%f %f %f\n",ef[0][i],ef[1][i],x_pow[i]);
    ef[0][i] /= (x_pow[i] + 1e-10f);
    ef[1][i] /= (x_pow[i] + 1e-10f);
    abs_ef = sqrtf(ef[0][i] * ef[0][i] + ef[1][i] * ef[1][i]);
    //printf("2.%f %f %f\n",ef[0][i],ef[1][i],abs_ef);

    // abs_ef太大表示误差信号太大，很有可能滤波器估算的不�?
    // 当一段时刻处于回声消除状态的时�?�，回声消除估算的可能比较准了，abs_ef就会变小
    // 这个时�?�abs_ef突然变大�?如果根据ef来调整步长，步长就会太大，反而影响目前稳定的回声消除状�??
    // 当双讲的时�?�abs_ef大概率会触发到error_threshold，调整的过大可能会影响双讲，太慢又容易噪声回声泄�?
    // error_threshold还有另外�?个作用，当远端信号为接近0时，近端信号很小，这样滤波器的系数可能会很大�?
    // �?旦下�?个远端信号声音变的大�?些，乘以这个滤波器系数就会导致滤波器输出的溢�?
    // �?般来说，光靠pbfdaf是很难将回声信号估算的比较准，所以这个误差一般会很大，所以error_threshold�?般会触发
    if (abs_ef > error_threshold) {
      //让步长的调整尽量靠近error_threshold
      abs_ef = error_threshold / (abs_ef + 1e-10f);
      //printf("3.%f\n",abs_ef);
      //ef尽量接近error_threshold
      ef[0][i] *= abs_ef;
      ef[1][i] *= abs_ef;
    }

    // Stepsize factor
    // mu默认值是0.5 适当调大对双讲有帮助 太大会引起回声消除不稳定
    ef[0][i] *= mu;
    ef[1][i] *= mu;
    //printf("4.%f %f\n",ef[0][i],ef[1][i]);
  }
}

static void ScaleErrorSignal(float mu,
                             float error_threshold,
                             float x_pow[PART_LEN1],
                             float ef[2][PART_LEN1]) {
  #if defined(MIPS_FPU_LE)
  ScaleErrorSignal_MIPS(mu,error_threshold,x_pow,ef);
  #elif defined(ARCH_ARM_NEON)
  ScaleErrorSignal_NEON(mu,error_threshold,x_pow,ef);
  #elif defined(ARCH_X86_FAMILY)
  if (use_sse2_) {
    ScaleErrorSignal_SSE2(mu,error_threshold,x_pow,ef);
  } 
  else {
    ScaleErrorSignal_C(mu,error_threshold,x_pow,ef);
  }
  #else
  ScaleErrorSignal_C(mu,error_threshold,x_pow,ef);
  #endif
}

/*
 * Stage1 步骤③：频域 LMS 滤波器系数更新
 *
 * PBFDAF 系数更新规则（约束时域 LMS，带 overlap-save 窗约束）：
 *   对每个分区 k：
 *     grad_k = IFFT(X_k* · E)   （频域互相关，取前 64 点，后 64 点清零）
 *     H_k   += FFT(grad_k)       （重新变换回频域，叠加到滤波器系数）
 *
 * 为什么要 IFFT→清后半段→再 FFT（而不是直接 H += X*·E）？
 *   频域直接相乘等价于时域循环卷积，长度会超出 64 点产生时域混叠。
 *   通过 IFFT→清后半段（强制因果约束）→FFT，保证等价于线性卷积，
 *   这是 overlap-save PBFDAF 的标准约束步骤。
 */
static void FilterAdaptation_C(
    int num_partitions,
    int x_fft_buf_block_pos,
    float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
    float e_fft[2][PART_LEN1],
    float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]) {
  int i, j;
  float fft[PART_LEN2];
  for (i = 0; i < num_partitions; i++) {
    int xPos = (i + x_fft_buf_block_pos) * (PART_LEN1);
    int pos;
    // Check for wrap
    if (i + x_fft_buf_block_pos >= num_partitions) {
      xPos -= num_partitions * PART_LEN1;
    }

    pos = i * PART_LEN1; //滤波器从�?0组开始，不受x_fft_buf_block_pos影响

    // x_fft_buf 远端信号共轭*误差信号 X'*E  共轭：虚部取�?
    for (j = 0; j < PART_LEN; j++) {
      fft[2 * j] = MulRe(x_fft_buf[0][xPos + j], -x_fft_buf[1][xPos + j], e_fft[0][j], e_fft[1][j]);
      fft[2 * j + 1] = MulIm(x_fft_buf[0][xPos + j], -x_fft_buf[1][xPos + j], e_fft[0][j], e_fft[1][j]);
    }
    // 按照逆傅立叶变换的格式准备数�?
    fft[1] = MulRe(x_fft_buf[0][xPos + PART_LEN], -x_fft_buf[1][xPos + PART_LEN],e_fft[0][PART_LEN], e_fft[1][PART_LEN]);

    // X'*E 逆傅立叶变换
    OouraFft_InverseFft(fft);
    //取前面的64个时域数�? 后面的清0
    memset(fft + PART_LEN, 0, sizeof(float) * PART_LEN);

    // fft scaling
    // 逆傅立叶变换获得正确的�?? �?要进行缩�?
    {
      float scale = 2.0f / PART_LEN2;
      for (j = 0; j < PART_LEN; j++) {
        fft[j] *= scale;
      }
    }
    //傅立叶变�?
    OouraFft_Fft(fft);

    //更新滤波器系�?
    h_fft_buf[0][pos] += fft[0];
    h_fft_buf[0][pos + PART_LEN] += fft[1];

    for (j = 1; j < PART_LEN; j++) {
      h_fft_buf[0][pos + j] += fft[2 * j];
      h_fft_buf[1][pos + j] += fft[2 * j + 1];
    }
  }
}

static void FilterAdaptation(
    int num_partitions,
    int x_fft_buf_block_pos,
    float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
    float e_fft[2][PART_LEN1],
    float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]) {               
  #if defined(MIPS_FPU_LE)
  FilterAdaptation_MIPS(num_partitions,x_fft_buf_block_pos,x_fft_buf,e_fft,h_fft_buf);
  #elif defined(ARCH_ARM_NEON)
  FilterAdaptation_NEON(num_partitions,x_fft_buf_block_pos,x_fft_buf,e_fft,h_fft_buf);
  #elif defined(ARCH_X86_FAMILY)
  if (use_sse2_) {
    FilterAdaptation_SSE2(num_partitions,x_fft_buf_block_pos,x_fft_buf,e_fft,h_fft_buf);
  } 
  else {
    FilterAdaptation_C(num_partitions,x_fft_buf_block_pos,x_fft_buf,e_fft,h_fft_buf);
  }
  #else
  FilterAdaptation_C(num_partitions,x_fft_buf_block_pos,x_fft_buf,e_fft,h_fft_buf);
  #endif
}

static int PartitionDelay_C(
    int num_partitions,
    float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]) {
  // Measures the energy in each filter partition and returns the partition with
  // highest energy.
  // TODO(bjornv): Spread computational cost by computing one partition per
  // block?
  float wfEnMax = 0;
  int i;
  int delay = 0;

  for (i = 0; i < num_partitions; i++) {
    int j;
    int pos = i * PART_LEN1;
    float wfEn = 0;
    //每组滤波器的 65个复数系数求平方�?
    for (j = 0; j < PART_LEN1; j++) {
      wfEn += h_fft_buf[0][pos + j] * h_fft_buf[0][pos + j] + h_fft_buf[1][pos + j] * h_fft_buf[1][pos + j];
    }
    if (wfEn > wfEnMax) {
      wfEnMax = wfEn;
      delay = i;
    }
  }
  return delay;
}

static int PartitionDelay(
    int num_partitions,
    float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]) {
  #if defined(MIPS_FPU_LE)
  return PartitionDelay_C(num_partitions,h_fft_buf);
  #elif defined(ARCH_ARM_NEON)
  return PartitionDelay_NEON(num_partitions,h_fft_buf);
  #elif defined(ARCH_X86_FAMILY)
  if (use_sse2_) {
    return PartitionDelay_SSE2(num_partitions,h_fft_buf);
  } 
  else {
    return PartitionDelay_C(num_partitions,h_fft_buf);
  }
  #else
  return PartitionDelay_C(num_partitions,h_fft_buf);
  #endif
}

// Updates the following smoothed Power Spectral Densities (PSD):
//  - sd  : near-end
//  - se  : residual echo
//  - sx  : far-end
//  - sde : cross-PSD of near-end and residual echo
//  - sxd : cross-PSD of near-end and far-end
//
// In addition to updating the PSDs, also the filter diverge state is
// determined.
/*
 * Stage2 步骤①：更新功率谱与互功率谱（指数平滑）
 *
 * 更新以下 5 个量（α = ptrGCoh[0]，β = ptrGCoh[1]，α+β=1）：
 *   sd[i]    = α·sd[i]    + β·|D(f)|²          （近端功率谱）
 *   se[i]    = α·se[i]    + β·|E(f)|²          （误差功率谱）
 *   sx[i]    = α·sx[i]    + β·max(|X(f)|²,kMin) （远端功率谱，下界保护）
 *   sde[i]   = α·sde[i]   + β·D(f)·E*(f)       （近端-误差互功率谱）
 *   sxd[i]   = α·sxd[i]   + β·D(f)·X*(f)       （远端-近端互功率谱）
 *
 * 平滑系数选取：
 *   mult=1（8kHz）  → α=0.90，β=0.10
 *   mult≥2（16/48kHz）→ α=0.93，β=0.07（更平滑，适合更高采样率）
 *
 * 滤波器发散检测：
 *   若 seSum > sdSum     → 误差能量超过近端能量，标记发散（输出近端原信号）
 *   若 seSum > 20×sdSum  → 严重发散，清零滤波器系数重新收敛
 */
static void UpdateCoherenceSpectra_C(int mult,
                                   float efw[2][PART_LEN1],
                                   float dfw[2][PART_LEN1],
                                   float xfw[2][PART_LEN1],
                                   CoherenceState* coherence_state,
                                   short* filter_divergence_state,
                                   int* extreme_filter_divergence) {
  // Power estimate smoothing coefficients.
  const float* ptrGCoh = kNormalSmoothingCoefficients[mult - 1];
  int i;
  float sdSum = 0, seSum = 0;
  
  //ptrGCoh[0] 0.93 ptrGCoh[1] 0.07 因为远端信号和近端信号有延时，所以不能直接计算互相关
  for (i = 0; i < PART_LEN1; i++) {
    // sd 近端信号的能量平�?
    coherence_state->sd[i] = ptrGCoh[0] * coherence_state->sd[i] + ptrGCoh[1] * (dfw[0][i] * dfw[0][i] + dfw[1][i] * dfw[1][i]);
    // se 误差信号的能量平�?
    coherence_state->se[i] = ptrGCoh[0] * coherence_state->se[i] + ptrGCoh[1] * (efw[0][i] * efw[0][i] + efw[1][i] * efw[1][i]);
    // We threshold here to protect against the ill-effects of a zero farend.
    // The threshold is not arbitrarily chosen, but balances protection and
    // adverse interaction with the algorithm's tuning.
    // TODO(bjornv): investigate further why this is so sensitive.
    // sx 远端信号的能量平�?
    coherence_state->sx[i] = ptrGCoh[0] * coherence_state->sx[i] + ptrGCoh[1] * SPL_MAX(xfw[0][i] * xfw[0][i] + xfw[1][i] * xfw[1][i], kMinFarendPSD);

    //sde 近端信号和误差信号的互相�? 频域相乘
    coherence_state->sde[i][0] = ptrGCoh[0] * coherence_state->sde[i][0] + ptrGCoh[1] * (dfw[0][i] * efw[0][i] + dfw[1][i] * efw[1][i]);
    coherence_state->sde[i][1] = ptrGCoh[0] * coherence_state->sde[i][1] + ptrGCoh[1] * (dfw[0][i] * efw[1][i] - dfw[1][i] * efw[0][i]);

    //sxd 近端信号和远端信号的互相�? 频域相乘
    coherence_state->sxd[i][0] = ptrGCoh[0] * coherence_state->sxd[i][0] + ptrGCoh[1] * (dfw[0][i] * xfw[0][i] + dfw[1][i] * xfw[1][i]);
    coherence_state->sxd[i][1] = ptrGCoh[0] * coherence_state->sxd[i][1] + ptrGCoh[1] * (dfw[0][i] * xfw[1][i] - dfw[1][i] * xfw[0][i]);

    //近端信号平滑能量�?
    sdSum += coherence_state->sd[i];
    //误差信号平滑能量�?
    seSum += coherence_state->se[i];    
  }

  // Divergent filter safeguard update.
  // 当误差信号的能量�?>近端信号能量和的时�?�，则认为不合理 因为近端能量�?-误差能量和为负的 这个时�?�直接采用近端信号做输出
  *filter_divergence_state = (*filter_divergence_state ? 1.05f : 1.0f) * seSum > sdSum;

  // Signal extreme filter divergence if the error is significantly larger than the nearend (13 dB).
  // 当误差信号能量和 > 19.95 * 近端信号能量�? 滤波器系数需要清�? 重新�?�?
  *extreme_filter_divergence = (seSum > (19.95f * sdSum));
}

static void UpdateCoherenceSpectra(int mult,
                                   float efw[2][PART_LEN1],
                                   float dfw[2][PART_LEN1],
                                   float xfw[2][PART_LEN1],
                                   CoherenceState* coherence_state,
                                   short* filter_divergence_state,
                                   int* extreme_filter_divergence) {
  #if defined(MIPS_FPU_LE)
  UpdateCoherenceSpectra_C(mult,efw,dfw,xfw,coherence_state,filter_divergence_state,extreme_filter_divergence);
  #elif defined(ARCH_ARM_NEON)
  UpdateCoherenceSpectra_NEON(mult,efw,dfw,xfw,coherence_state,filter_divergence_state,extreme_filter_divergence);
  #elif defined(ARCH_X86_FAMILY)
  if (use_sse2_) {
    UpdateCoherenceSpectra_SSE2(mult,efw,dfw,xfw,coherence_state,filter_divergence_state,extreme_filter_divergence);
  } 
  else {
    UpdateCoherenceSpectra_C(mult,efw,dfw,xfw,coherence_state,filter_divergence_state,extreme_filter_divergence);
  }
  #else
  UpdateCoherenceSpectra_C(mult,efw,dfw,xfw,coherence_state,filter_divergence_state,extreme_filter_divergence);
  #endif
}

/*
 * Stage2 步骤②：计算子带相干度
 *
 * 相干度定义（取值 0~1）：
 *   cohde[i] = |sde[i]|² / (sd[i] · se[i])
 *            = 近端与误差信号的归一化互相关能量
 *     → 趋近 1：近端与误差高度相似，说明滤波器没有消除回声（近端说话或滤波器未收敛）
 *     → 趋近 0：近端与误差不相似，说明滤波器已有效消除了回声（纯回声场景）
 *
 *   cohxd[i] = |sxd[i]|² / (sx[i] · sd[i])
 *            = 远端与近端信号的归一化互相关能量
 *     → 趋近 1：远端与近端高度相似，说明近端主要是回声
 *     → 趋近 0：远端与近端不相似，说明近端是本地语音（或静音）
 *
 * 这两个相干度是 Stage2 状态判断和增益计算的核心输入。
 */
static void ComputeCoherence_C(const CoherenceState* coherence_state,
                             float* cohde,
                             float* cohxd) {
  // Subband coherence
  // sde 近端信号和误差信号的互相�? 频域相乘
  // sxd 近端信号和远端信号的互相�? 频域相乘
  // sd 近端信号的能量平�?
  // se 误差信号的能量平�?
  // sx 远端信号的能量平�?
  int i;
  for (i = 0; i < PART_LEN1; i++) {
    // cohde 近端信号和误差信号的互相关能�?/(近端信号能量*误差信号能量)
    cohde[i] = (coherence_state->sde[i][0] * coherence_state->sde[i][0] + coherence_state->sde[i][1] * coherence_state->sde[i][1]) /
                                                                    (coherence_state->sd[i] * coherence_state->se[i] + 1e-10f);
    // cohxd 近端信号和远端信号的互相关能�?/(远端信号能量*近端信号的能�?)
    cohxd[i] = (coherence_state->sxd[i][0] * coherence_state->sxd[i][0] + coherence_state->sxd[i][1] * coherence_state->sxd[i][1]) /
                                                                    (coherence_state->sx[i] * coherence_state->sd[i] + 1e-10f);
  }
}

static void ComputeCoherence(const CoherenceState* coherence_state,
                             float* cohde,
                             float* cohxd) {
  #if defined(MIPS_FPU_LE)
  ComputeCoherence_C(coherence_state,cohde,cohxd);
  #elif defined(ARCH_ARM_NEON)
  ComputeCoherence_NEON(coherence_state,cohde,cohxd);
  #elif defined(ARCH_X86_FAMILY)
  if (use_sse2_) {
    ComputeCoherence_SSE2(coherence_state,cohde,cohxd);
  } 
  else {
    ComputeCoherence_C(coherence_state,cohde,cohxd);
  }
  #else
  ComputeCoherence_C(coherence_state,cohde,cohxd);
  #endif  
}

/*
 * Stage2 步骤④：非线性 Overdrive（对抑制增益做指数放大）
 *
 * hNl[i] 的原始值由 FormSuppressionGain 计算，范围 [0,1]。
 * 直接用 hNl 乘以误差信号往往残余回声仍较多，需要进一步压制。
 *
 * 处理分两步：
 *
 * 步骤 A：子带加权（高频额外压制）
 *   对 hNl[i] > hNlFb 的频带（回声泄漏较多的频带）：
 *     hNl[i] = weightCurve[i]·hNlFb + (1-weightCurve[i])·hNl[i]
 *   weightCurve 随频率升高而增大（i=0 为 0，i=64 约 0.4），
 *   高频受更强拉低，符合人声高频弱、高频回声可更激进压制的特点。
 *
 * 步骤 B：指数放大
 *   hNl[i] = hNl[i] ^ (overdrive_scaling · overDriveCurve[i])
 *   由于 hNl[i] ∈ [0,1]，指数 > 1 时 hNl 会变得更小（压制更强）。
 *   overDriveCurve 随频率升高（从 1.0 到 2.0），高频衰减比低频更快。
 *
 * 步骤 C：全频带门限（调试参数）
 *   若低频段（i=2~14）的 hNl 总和 < 1.25，或全频段 < 2.5，
 *   则认为处于强回声状态，将所有 hNl 清零（全频带压制）。
 *
 * overdrive_scaling 由 FormSuppressionGain 动态更新，越大压制越强。
 */
static void Overdrive_C(float overdrive_scaling,
                        const float hNlFb,
                        float hNl[PART_LEN1]) {

  // jay
  // 现在的情况是双讲的时候，overdrive_scaling压制的越多，不利于双�?
  // hNlFb越大的时候，overdrive_scaling�?要越接近1，近端说话概率大，overdrive_scaling不能小于1
  // hNlFb越小的时候，overdrive_scaling�?要越大，回声概率�?
  // printf("%f %f\n",hNlFb,overdrive_scaling);
  #if 0
  float overdrive_scaling_modify = 1.0f + -(float)(log(hNlFb + 1e-10f));
  //printf("%f %f\n",hNlFb,overdrive_scaling_modify);
  #endif
  int i = 0;
  float hNlSum = 0.0f;
  float hNlSum_2_14 = 0.0f;

  for (i = 0; i < PART_LEN1; ++i) {
    // Weight subbands
    // hNlFb表示hNl的较小�??
    #if 1
    if (hNl[i] > hNlFb) {
      // 超过了基准�?�，意味�?可能存在回声泄漏 那么�?要多压制�?�?
      // weightCurve 频率越高，压制多�?些，可以多压制一些高�?
      // 双讲判断�?2�?14频带计算出来的，高频可能不准，尽量多�? 并且人声的高频段比较�? 在�?�话的场景，高频段可以多压制�?�?
      // 当i�?0�? weightCurve = 0    hNl[i]保持不变
      // 当i�?1�? weightCurve = 0.1  hNl[i] = 0.1*hNlFb + 0.9*hNl[i]   //减少�?�?
      // 当i�?2�? weightCurve = 0.13 hNl[i] = 0.13*hNlFb + 0.87*hNl[i] //再多减一�?
      hNl[i] = weightCurve[i] * hNlFb + (1 - weightCurve[i]) * hNl[i];
    }
    #endif
    // hNl[i]的overdrive_scaling * overDriveCurve[i]次方
    // hNl[i]的�?�是�?直小�?1
    // 如果overdrive_scaling * overDriveCurve[i] > 1，hNl会变�?
    // overDriveCurve在高频段的�?�更大，意味�?hNl衰减的更�?
    // before hNl[i] 0.487795 overdrive_scaling * overDriveCurve[i] 6.420938
    // after: hNl[i] 0.009958
    // 感觉hNl[i]在这里会变的更小
    
    // 500HZ以下不做再次压制，因�?500HZ以下的相似度已经很高�?(可能是谐�?))
    //if(i > 4)
      hNl[i] = powf(hNl[i], overdrive_scaling * overDriveCurve[i]);
      //hNl[i] = powf(hNl[i], overdrive_scaling_modify * overDriveCurve[i]);

    // 高频�? 太小�? 直接丢掉 这个参数值得调整
    #if 0
    if(i > 14){
      if(hNl[i] < 0.2f){
        hNl[i] = 0.0f;
        }
    }
    #endif

    hNlSum += hNl[i];
    
    if(i >= 2 && i <= 14){
      hNlSum_2_14 += hNl[i];
    }
  }

  //hNl[0] = 0;
  //hNl[1] = 0;
  
  //printf("%f %f\n",hNlSum_2_14,hNlSum);

  #if 1 //这个参数值得调整，直接影响回�?
  // if(hNlSum_2_14 < 0.8f){
  if(hNlSum_2_14 < 1.25f){
      for (i = 0; i < PART_LEN1; ++i) {
        hNl[i] = 0;
      }
  }
  #endif

  #if 1
  // if(hNlSum < 2.0f){
  if(hNlSum < 2.5f){
      for (i = 0; i < PART_LEN1; ++i) {
        hNl[i] = 0;
      }
  }
  #endif

  #if 0
  if(hNlSum_2_14 > 1.0f){
    printf("%f\n",hNlSum_2_14);
  }
  #endif
}

static void Overdrive(float overdrive_scaling,
                      const float hNlFb,
                      float hNl[PART_LEN1]) {
  #if 0
  #if defined(MIPS_FPU_LE)
  Overdrive_MIPS(overdrive_scaling,hNlFb,hNl);
  #elif defined(ARCH_ARM_NEON)
  Overdrive_NEON(overdrive_scaling,hNlFb,hNl);
  #elif defined(ARCH_X86_FAMILY)
  if (use_sse2_) {
    Overdrive_SSE2(overdrive_scaling,hNlFb,hNl);
  } 
  else {
    Overdrive_C(overdrive_scaling,hNlFb,hNl);
  }
  #else
  Overdrive_C(overdrive_scaling,hNlFb,hNl);
  #endif
  #endif
  Overdrive_C(overdrive_scaling,hNlFb,hNl);
}

static void Suppress_C(const float hNl[PART_LEN1], float efw[2][PART_LEN1]) {
  int i;
  for (i = 0; i < PART_LEN1; ++i) {
    // Suppress error signal
    efw[0][i] *= hNl[i];
    efw[1][i] *= hNl[i];

    // Ooura fft returns incorrect sign on imaginary component. It matters here
    // because we are making an additive change with comfort noise.
    efw[1][i] *= -1;
  }
}

static void Suppress(const float hNl[PART_LEN1], float efw[2][PART_LEN1]) {
  #if defined(MIPS_FPU_LE)
  Suppress_MIPS(hNl,efw);
  #elif defined(ARCH_ARM_NEON)
  Suppress_NEON(hNl,efw);
  #elif defined(ARCH_X86_FAMILY)
  if (use_sse2_) {
    Suppress_SSE2(hNl,efw);
  } 
  else {
    Suppress_C(hNl,efw);
  }
  #else
  Suppress_C(hNl,efw);
  #endif
}

/*
 * Stage1 主函数：线性回声消除（PBFDAF-NLMS）
 *
 * 输入：
 *   x_fft     : 当前帧远端信号 FFT（实/虚拼接，2×65 = 130 个 float）
 *   y         : 当前帧近端时域信号（64 点）
 *   x_pow     : 远端平滑功率谱（65 点，NLMS 归一化分母）
 *   h_fft_buf : 自适应滤波器系数（12 分区频域，双精度写入）
 * 输出：
 *   echo_subtractor_output : 线性消除后的误差信号 e = d - ŝ（64 点）
 *
 * 执行顺序：
 *   ① 更新远端 FFT 环形缓冲（xfBuf），写入最新帧
 *   ② 严重发散时清零滤波器
 *   ③ FilterFar：利用 12 分区历史计算回声估计 ŝ(频域) → IFFT → 时域 s
 *   ④ e = y - s（时域相减）
 *   ⑤ ScaleErrorSignal：NLMS 归一化 + 限幅 + 乘步长 → 滤波器更新量
 *   ⑥ FilterAdaptation：H += ΔH（时域约束 + 重新 FFT）
 *   ⑦ 将 e 写入 echo_subtractor_output 供 Stage2 使用
 */
static void EchoSubtraction(int num_partitions,
                            int* extreme_filter_divergence,
                            float filter_step_size,
                            float error_threshold,
                            float* x_fft,
                            int* x_fft_buf_block_pos,
                            float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                            float* y,
                            float x_pow[PART_LEN1],
                            float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                            float echo_subtractor_output[PART_LEN]){
  int i;                            
  float s_fft[2][PART_LEN1];
  float s_extended[PART_LEN2];
  float e_extended[PART_LEN2];
  float* s;
  float e[PART_LEN];
  float e_fft[2][PART_LEN1];

  // Update the x_fft_buf block position.
  // num_partitions 12 滤波器组�?
  // 远端信号也需要保存最近的12�?
  // *x_fft_buf_block_pos 初始值为0
  (*x_fft_buf_block_pos)--;
  if ((*x_fft_buf_block_pos) == -1) {
    *x_fft_buf_block_pos = num_partitions - 1;
  }

  // Buffer x_fft.
  // x_fft 2*65
  // 实部 ----------- 65
  // 虚部 ----------- 65
  // x_fft_buf 2*�?12*65�?
  // 保存当前的远端复数帧到远端buffer的最�?
  memcpy(x_fft_buf[0] + (*x_fft_buf_block_pos) * PART_LEN1, x_fft, sizeof(float) * PART_LEN1);
  memcpy(x_fft_buf[1] + (*x_fft_buf_block_pos) * PART_LEN1, &x_fft[PART_LEN1], sizeof(float) * PART_LEN1);

  memset(s_fft, 0, sizeof(s_fft));

  // Conditionally reset the echo subtraction filter if the filter has diverged significantly.
  // 滤波器系数清0
  if (*extreme_filter_divergence) {
    memset(h_fft_buf, 0, 2 * kNormalNumPartitions * PART_LEN1 * sizeof(h_fft_buf[0][0]));
    *extreme_filter_divergence = 0;
  }

  // Produce echo estimate s_fft.
  // s_fft = x_fft_buf * h_fft_buf 12组滤波器和远端信号的乘加
  FilterFar(num_partitions, *x_fft_buf_block_pos, x_fft_buf, h_fft_buf, s_fft);

  // Compute the time-domain echo estimate s.
  // 缩放后的逆傅立叶变换 为什么要缩放
  // FFT和ScaledInverseFft是一对傅立叶变换和�?�傅立叶变换计算 缩放的系�?2.0恰好可以还原数据
  ScaledInverseFft(s_fft, s_extended, 2.0f, 0);
  //后半�?64字节时域信号
  s = &s_extended[PART_LEN];

  // Compute the time-domain echo prediction error.
  // 近端信号-回声估计信号
  for (i = 0; i < PART_LEN; ++i) {
    e[i] = y[i] - s[i];
  }
  
  // Compute the frequency domain echo prediction error.
  // 前面�?64�?0，误差信号再次转换到频域e_fft
  memset(e_extended, 0, sizeof(float) * PART_LEN);
  memcpy(e_extended + PART_LEN, e, sizeof(float) * PART_LEN);
  Fft(e_extended, e_fft);
  
  // Scale error signal inversely with far power.
  // filter_step_size 0.5
  // error_threshold 1.5e-6f
  // xPow[i] = 0.9*xPow[i] + 0.1*12*far_spectrum 远端信号能量相关
  // 对e_fft进行修正并乘以步�? 得到滤波器系数调整�??
  ScaleErrorSignal(filter_step_size, error_threshold, x_pow, e_fft);
  // 12组滤波器系数更新 多滤波器会引起信号出现奇怪的频谱 单滤波器不会
  FilterAdaptation(num_partitions, *x_fft_buf_block_pos, x_fft_buf, e_fft, h_fft_buf);

  //echo_subtractor_output 保存为时域误差信�? 长度�?64
  memcpy(echo_subtractor_output, e, sizeof(float) * PART_LEN);
}

/*
 * Stage2 步骤③：根据相干度生成 NLP 抑制增益 hNl[i] ∈ [0,1]
 *
 * 状态判断依据（基于 2~14 频带的平均相干度）：
 *
 *   hNlDeAvg = mean(cohde[2..14])   近端-误差相干度均值
 *   hNlXdAvg = 1 - mean(cohxd[2..14])  近端-远端不相似度均值
 *
 *   ┌─────────────────┬───────────────┬────────────────┐
 *   │     状态        │  hNlDeAvg     │   hNlXdAvg     │
 *   ├─────────────────┼───────────────┼────────────────┤
 *   │ 只有近端说话    │  ≈ 1（很相似）│  ≈ 1（不相似） │
 *   │ 双讲            │  中等         │   中等          │
 *   │ 纯回声          │  ≈ 0（不相似）│  ≈ 0（很相似） │
 *   └─────────────────┴───────────────┴────────────────┘
 *
 * hNl 计算规则（按状态分支）：
 *   hNlXdAvgMin == 1（长期无回声）:
 *     近端说话 → hNl = cohde（几乎不压制）
 *     其他     → hNl = 1 - cohxd（中等压制）
 *   hNlXdAvgMin < 1（曾检测到回声）:
 *     近端说话 → hNl = cohde（不压制）
 *     回声/双讲→ hNl = min(cohde, 1-cohxd)（取更保守的值）
 *
 * overdrive 更新逻辑：
 *   hNlFbLow（hNl 低频中位数）< 0.6 时更新 hNlFbMin 历史最小值；
 *   发现新最小值后的第 2 次迭代，按目标压制量重新计算 overDrive：
 *     overDrive = max(kTargetSupp[mode] / log(hNlFbMin), kNormalMinOverDrive[mode])
 *   overdrive_scaling 以 0.99/0.9 的惯性系数平滑跟踪 overDrive。
 */
static void FormSuppressionGain(AECV2* aecv2,
                                float cohde[PART_LEN1],
                                float cohxd[PART_LEN1],
                                float hNl[PART_LEN1]) {
  int i;
  float hNlDeAvg, hNlXdAvg;
  float hNlPref[kPrefBandSize];
  float hNlFb = 0, hNlFbLow = 0;
  const int prefBandSize = kPrefBandSize / aecv2->mult; // kPrefBandSize 24 prefBandSize 12
  const float prefBandQuant = 0.9f, prefBandQuantLow = 0.5f;
  const int minPrefBand = 4 / aecv2->mult;  //minPrefBand 2
  // Power estimate smoothing coefficients.
  const float* min_overdrive = kNormalMinOverDrive;

  //因为结构、声学等关系，导致完全回声消除状态时，远端信号和近端信号其实并没有那么相�?
  //�?以导致滤波器的估算结果也没那么准，所以完全回声消除状态，近端信号和误差信号也没有那么的不相似，还有残�?
 
  // cohxd 近端信号和远端信号的互相关能�?/(远端信号能量*近端信号的能�?)
  hNlXdAvg = 0;
  // i 2 ~ 14
  // 为什么只选择2�?14频段 人声�?丰富的频段范�?
  for (i = minPrefBand; i < prefBandSize + minPrefBand; ++i) {
    hNlXdAvg += cohxd[i];
  }

  hNlXdAvg /= prefBandSize;  //越大说明远端信号和近端信号越相似
  hNlXdAvg = 1 - hNlXdAvg;   //越小说明远端信号和近端信号越相似

  // cohde 近端信号和误差信号的互相关能�?/(近端信号能量*误差信号能量)
  hNlDeAvg = 0;
  for (i = minPrefBand; i < prefBandSize + minPrefBand; ++i) {
    hNlDeAvg += cohde[i];
  }

  hNlDeAvg /= prefBandSize; //越大说明近端信号和误差信号越相似 更有可能是双�? 也有可能是近端单�?

  // hNlXdAvg < 0.75f表示远端信号和近端信号往相似的方�? 处于双讲或回声消除状�?
  // 保存hNlXdAvg�?小�?? 越小越有可能是回声消除状�?
  if (hNlXdAvg < 0.75f && hNlXdAvg < aecv2->hNlXdAvgMin) {
    aecv2->hNlXdAvgMin = hNlXdAvg;
  }

   //printf("hNlDeAvg:%f, hNlXdAvg:%f\n",hNlDeAvg,hNlXdAvg);
  // 只有近端在说�? ：近端信号和误差信号非常的相�? 远端信号和近端信号非常的不相�? 
  // 双讲         ：近端信号和误差信号处于相似和不相似的中�? 远端信号和近端信号处于相似和不相似的中间 
  // 回声消除状�??  ：近端信号和误差信号非常的不相似 远端信号和近端信号非常的相似
  // Bug：在M220上，�?么也不播放的时�?�，参�?�信号也不会完全�?0，因为ADC器件总会带来噪声
  // 在这个时候，近端轻声说话或远距离说话，hNlDeAvg�?般会大于0.98，但是hNlXdAvg会在0.8之间徘徊
  // 这个时�?�，会进入stNearState = 0的状态，引发了回声消除，这个时�?�需要将hNlXdAvg < 0.8f调整到hNlXdAvg < 0.75f，防止误进入回声消除
  if (hNlDeAvg > 0.98f && hNlXdAvg > 0.9f) {
    // hNlDeAvg > 0.98f  表示近端信号和误差信号很相似
    // hNlXdAvg > 0.9f   表示远端信号和近端信号很不相�?
    aecv2->stNearState = 1;  //只有近端在说�?
  }
  else if (hNlDeAvg < 0.95f || hNlXdAvg < 0.8f) {
    // hNlDeAvg < 0.95f 表示近端信号和误差信号可能部分相似也可能很不相似
    // hNlXdAvg < 0.8f  表示近端信号和误差信号可能相似也有可能非常相�?
    aecv2->stNearState = 0; //肯定不是只有近端在说话的状�?�，回声消除状�?�或双讲状�??
  }

  // 也就是hNlXdAvg会比较大 此时可能是只有近端在说话的状�?
  // hNlXdAvgMin在后面每次循环会微小的增加，如果加到1，那么这�?段时间就认为只有近端在说话的概率比较�?
  // 在远端信号存在的时�?�，hNlXdAvgMin�?1的概率不�?
  if (aecv2->hNlXdAvgMin == 1) {
    aecv2->echoState = 0; //此时不处于回声消除状�?
    aecv2->overDrive = min_overdrive[aecv2->nlp_mode];  //{1.0f, 2.0f, 5.0f};

    // 处于只有近端在说话的状�?? 近端信号和误差信号非常的相似
    if (aecv2->stNearState == 1) {
      // cohde此时会比较大 也就是尽量不做压�?
      memcpy(hNl, cohde, sizeof(hNl[0]) * PART_LEN1);
      hNlFb = hNlDeAvg;       // hNlDeAvg此时会比较大 压制的少�?�?
      hNlFbLow = hNlDeAvg;
    }
    else {
      // 在这个时候cohxd会接�?0，所�?1 - cohxd[i]会接�?1，压制的并不�?
      for (i = 0; i < PART_LEN1; ++i) {
        hNl[i] = 1 - cohxd[i];
        hNl[i] = SPL_MAX(hNl[i], 0.f); //max()
      }
      hNlFb = hNlXdAvg;    // hNlXdAvg这个时�?�会比较�? 压制的多�?�?
      hNlFbLow = hNlXdAvg;
    }
  }
  // 否则有大概率处于回声消除状�?�或双讲状�??
  else {
    // 处于只有近端在说话的状�?? 近端信号和误差信号非常的相似
    if (aecv2->stNearState == 1) {
      aecv2->echoState = 0;
      memcpy(hNl, cohde, sizeof(hNl[0]) * PART_LEN1);
      hNlFb = hNlDeAvg;
      hNlFbLow = hNlDeAvg;
    } 
    else {
      // 处于回声消除状�?�或者双讲状�?
      aecv2->echoState = 1;
      for (i = 0; i < PART_LEN1; ++i) {
        hNl[i] = SPL_MIN(cohde[i], 1 - cohxd[i]);
        hNl[i] = SPL_MAX(hNl[i], 0.f); //max()
      }

      // 此时的操作和上面不太�?样了
      // Select an order statistic from the preferred bands.
      // TODO(peah): Using quicksort now, but a selection algorithm may be preferred.
      // hNlPref存储12组hNl�?
      memcpy(hNlPref, &hNl[minPrefBand], sizeof(float) * prefBandSize);
      // 12组hNL值进行排�?
      qsort(hNlPref, prefBandSize, sizeof(float), CmpFloat);
      // 选择�?个合适的hnl基准值hNlFb，防止某些频段回声多泄漏
      // 选择�?个中间偏大的hNL�?
      hNlFb = hNlPref[(int)(floor(prefBandQuant * (prefBandSize - 1)))];
      // 选择�?个中间的hNL�?
      hNlFbLow = hNlPref[(int)(floor(prefBandQuantLow * (prefBandSize - 1)))];
    }
  }

  // Track the local filter minimum to determine suppression overdrive.
  // hNlFbLocalMin存储hNlFbLow的最小�??
  // 测试发现hNlFbLow在只有近端信号存在时，大约是0.99，当处于回声消除状�?�时，有大概率会小于0.6  �?么都不播放的时�?�，仅仅空间噪声的时候，大约0.99
  // 在回声消除的时�?�，容易刷新overDrive的�??
  if (hNlFbLow < 0.6f && hNlFbLow < aecv2->hNlFbLocalMin) {
    aecv2->hNlFbLocalMin = hNlFbLow;
    aecv2->hNlFbMin = hNlFbLow;
    aecv2->hNlNewMin = 1;   //出现了新的更小的hNlFbLow�?
    aecv2->hNlMinCtr = 0;
  }
  
  //hNlFbLocalMin缓慢增加�?�? 否则hNlFbLocalMin会越来越�?
  aecv2->hNlFbLocalMin = SPL_MIN(aecv2->hNlFbLocalMin + 0.0008f / aecv2->mult, 1);
  //hNlXdAvgMin是不会大�?1�?
  //hNlXdAvgMin缓慢增加�?�? 否则hNlXdAvgMin会越来越�?
  aecv2->hNlXdAvgMin = SPL_MIN(aecv2->hNlXdAvgMin + 0.0006f / aecv2->mult, 1);

  if (aecv2->hNlNewMin == 1) {
    aecv2->hNlMinCtr++;
  }

  // 出现了更小的hNlFbLow值后的第二次更新overDrive
  if (aecv2->hNlMinCtr == 2) {
    aecv2->hNlNewMin = 0;
    aecv2->hNlMinCtr = 0;
    //static const float kTargetSupp[3] = {-6.9f, -11.5f, -18.4f};
    //static const float kNormalMinOverDrive[3] = {1.0f, 2.0f, 5.0f};
    //hNlFbMin   log     overDrive
    //0.112298 -2.186603 3.155580
    //0.256974 -1.358779 5.078088
    //hNlFbMin越小 overDrive越小
    //是否是hNl已经够小了，不需要多压制了？这里会引起双讲变�?
    //hNlFbMin越大表示双讲可能性越大，然后这里的设计overDrive反�?�更大了，也就是压制的越多，双讲变差�?
    //printf("%f\n",aecv2->hNlFbMin);
    
    //overDrive只有在回声消除的时�?�才刷新
    //设计变化更为�?
    //hNlFbMin越小，overDrive越大 
    //hNlFbMin越大，overDrive越小
    #if 1
    aecv2->overDrive = SPL_MAX(kTargetSupp[aecv2->nlp_mode] / (float)(log(aecv2->hNlFbMin + 1e-10f) + 1e-10f),
                                min_overdrive[aecv2->nlp_mode]);
    #endif

    #if 0
    aecv2->overDrive = 1.0f + -(float)(log(aecv2->hNlFbMin + 1e-10f));
    if(aecv2->overDrive < 1.0f){
      aecv2->overDrive = 1.0f;
    }
    #endif
  }

  // Smooth the overdrive.
  // �?旦进入回声消除状态，overDrive的�?�比较容易刷�?
  // 当什么也不做或�?�进入只有近端有声音时，hNlXdAvgMin == 1触发，overDrive的�?�得刷新�?1，这个时候overdrive_scaling会迅速接�?1
  // overdrive_scaling是渐进变化的，有助于混响的控�?
  // overdrive_scaling慢慢接近overdrive
  if (aecv2->overDrive < aecv2->overdrive_scaling) {
    // �?旦有进入回声消除状�?�overdrive_scaling很大，此时出现双讲overDrive就会变小，但是overdrive_scaling是缓慢下降的,会影响双�?
    // 但是下降快了，混响的控制就没有那么好
    // 实际发现overDrive的�?�并不太容易更新，进入回声消除的时�?�，突然发生双讲overDrive的�?�并不会刷新�?
    aecv2->overdrive_scaling = 0.99f * aecv2->overdrive_scaling + 0.01f * aecv2->overDrive;
  } 
  else {
    aecv2->overdrive_scaling = 0.9f * aecv2->overdrive_scaling + 0.1f * aecv2->overDrive;
  }

  // Apply the overdrive.
  // overdrive_scaling越大 hNl衰减的越�? 回声消除效果越好 但是双讲越差
  Overdrive(aecv2->overdrive_scaling, hNlFb, hNl);
}

static void GenerateComplexNoise(uint32_t* seed, float noise[2][PART_LEN1]) {
  size_t i;
  const float kPi2 = 6.28318530717959f;
  int16_t randW16[PART_LEN];
  SPL_RandUArray(randW16, PART_LEN, seed);

  noise[0][0] = 0;
  noise[1][0] = 0;
  for (i = 1; i < PART_LEN1; i++) {
    float tmp = kPi2 * randW16[i - 1] / 32768.f;
    noise[0][i] = cosf(tmp);
    noise[1][i] = -sinf(tmp);
  }
  noise[1][PART_LEN] = 0;
}

static void ComfortNoise(uint32_t* seed,
                         float e_fft[2][PART_LEN1],
                         const float* noise_spectrum,
                         const float* suppressor_gain) {
  int i;
  float complex_noise[2][PART_LEN1];

  GenerateComplexNoise(seed, complex_noise);

  // Shape, scale and add comfort noise.
  for (i = 1; i < PART_LEN1; ++i) {
    float noise_scaling = sqrtf(SPL_MAX(1 - suppressor_gain[i] * suppressor_gain[i], 0)) * sqrtf(noise_spectrum[i]);
    e_fft[0][i] += noise_scaling /2.0f * complex_noise[0][i];
    e_fft[1][i] += noise_scaling /2.0f * complex_noise[1][i];
  }
}

/*
 * Stage2 主函数：非线性回声抑制（NLP）
 *
 * 输入：
 *   nearend_overlap        : 近端信号 128 点（上帧 64 + 本帧 64，overlap-save）
 *   farend_overlap         : 远端信号 128 点（加窗 FFT 用于相干性分析）
 *   echo_subtractor_output : Stage1 输出的线性误差信号 e（64 点）
 * 输出：
 *   output : 最终回声消除结果（64 点，经 overlap-add 重建）
 *
 * 执行顺序：
 *   ① 三路信号加窗 FFT → efw（误差）、dfw（近端）、xfw（远端，延迟补偿）
 *   ② UpdateCoherenceSpectra：平滑功率谱 sd/se/sx 和互功率谱 sde/sxd
 *   ③ ComputeCoherence：计算 cohde 和 cohxd
 *   ④ divergeState 检测：发散时 efw = dfw（直接用近端）
 *   ⑤ FormSuppressionGain：状态机 → hNl（含 overdrive）
 *   ⑥ Suppress：efw *= hNl（频域乘法抑制）
 *   ⑦ ComfortNoise：注入舒适噪声（与 hNl 互补，保持自然感）
 *   ⑧ ScaledInverseFft：IFFT 还原时域
 *   ⑨ Overlap-Add 输出 + 饱和保护
 */
static void EchoSuppression(AECV2* aecv2,
                            float* nearend_overlap,
                            float farend_overlap[PART_LEN2],
                            float* echo_subtractor_output,
                            float output[PART_LEN]){
  int i;
  float fft[PART_LEN2];

  float efw[2][PART_LEN1];
  float xfw[2][PART_LEN1];
  float dfw[2][PART_LEN1];

  float* xfw_ptr = NULL;

  // Coherence and non-linear filter
  float cohde[PART_LEN1], cohxd[PART_LEN1];
  float hNl[PART_LEN1];

  // Filter energy
  const int delayEstInterval = 10 * aecv2->mult;

  // Update eBuf with echo subtractor output.
  // echo_subtractor_output 为误差信�?=近端信号-回声估算信号 64字节
  // 当前64字节的存储在aec->eBuf的后64字节�? overlap save
  memcpy(aecv2->eBuf + PART_LEN, echo_subtractor_output, sizeof(float) * PART_LEN);

  // Analysis filter banks for the echo suppressor.
  // Windowed near-end ffts.
  // 近端信号加窗 傅立叶变�?
  WindowData(fft, nearend_overlap);
  OouraFft_Fft(fft);
  // 近端信号傅立叶变换结�?128复数 保存在dfw�? 65复数
  StoreAsComplex(fft, dfw);

  // Windowed echo suppressor output ffts.
  // 误差信号加窗 傅立叶变�? 
  WindowData(fft, aecv2->eBuf);
  OouraFft_Fft(fft);
  // 误差信号傅立叶变换结�?128复数 保存在efw�? 65复数
  StoreAsComplex(fft, efw);

  // NLP

  // Convert far-end partition to the frequency domain with windowing.
  // 远端信号加窗 傅立叶变�?
  WindowData(fft, farend_overlap);
  // 远端信号 加窗傅立叶变换保存在xfw�? 65复数
  Fft(fft, xfw);
  xfw_ptr = &xfw[0][0];

  // Buffer far.
  // aec->xfwBuf 12�? 保存远端信号加窗傅立叶变换结果到缓存 65复数
  memcpy(aecv2->xfwBuf, xfw_ptr, sizeof(float) * 2 * PART_LEN1);

  //每隔20�? delayEstInterval=20
  aecv2->delayEstCtr++;
  if (aecv2->delayEstCtr == delayEstInterval) {
    aecv2->delayEstCtr = 0;
    // wfBuf为滤波器系数
    // 寻找12组滤波器中系数能量最大的索引
    // 因为在M220中，远端信号和近端信号有8ms的延时，�?以这里计算出来的值几乎为1
    // 因为在更新滤波器时，延时�?1的地方和近端信号�?接近，对应的滤波器系数也�?�?
    aecv2->delayIdx = PartitionDelay(aecv2->num_partitions, aecv2->wfBuf);
  }

  aecv2->delayIdx = 2;
  //printf("aecv2->delayIdx: %d\n",aecv2->delayIdx);

  // Use delayed far.
  // 找到能量�?大的滤波器组系数对应的远端信�? 保存到xfw
  // 为什么要寻找滤波器系数能量最大�?�对应的远端信号频谱呢？
  memcpy(xfw, aecv2->xfwBuf + aecv2->delayIdx * PART_LEN1, sizeof(xfw[0][0]) * 2 * PART_LEN1);

  //能量和互相关计算 保存在coherence_state
  UpdateCoherenceSpectra(aecv2->mult, efw, dfw, xfw, &aecv2->coherence_state,
                         &aecv2->divergeState, &aecv2->extreme_filter_divergence);

  // 求cohde cohxd
  ComputeCoherence(&aecv2->coherence_state, cohde, cohxd);

  // Select the microphone signal as output if the filter is deemed to have diverged.
  if (aecv2->divergeState) {
    //直接使用近端麦克风数�?
    memcpy(efw, dfw, sizeof(efw[0][0]) * 2 * PART_LEN1);
  }

  // 求supgain
  // 互相关的存在cohde/cohxd的�?�都小于1
  FormSuppressionGain(aecv2, cohde, cohxd, hNl);

  // E*hNl
  Suppress(hNl, efw);

  // Add comfort noise.
  ComfortNoise(&aecv2->seed, efw, aecv2->noisePow, hNl);

  // Inverse error fft.
  ScaledInverseFft(efw, fft, 2.0f, 1);

  // Overlap and add to obtain output.
  // 为什么前面是overlap save 结果变成了overlap add
  for (i = 0; i < PART_LEN; i++) {
    output[i] = (fft[i] * sqrtHanning[i] + aecv2->outBuf[i] * sqrtHanning[PART_LEN - i]);

    // Saturate output to keep it in the allowed range.
    output[i] = SPL_SAT(SPL_WORD16_MAX, output[i], SPL_WORD16_MIN);
  }

  //保存回声消除结果的后半段
  memcpy(aecv2->outBuf, &fft[PART_LEN], PART_LEN * sizeof(aecv2->outBuf[0]));

  // Copy the current block to the old position.
  memcpy(aecv2->eBuf, aecv2->eBuf + PART_LEN, sizeof(float) * PART_LEN);

  //xfwBuf向后平移�?个PART_LEN1 留出头部存放新的远端信号数据
  memmove(aecv2->xfwBuf + PART_LEN1, aecv2->xfwBuf, sizeof(aecv2->xfwBuf) - sizeof(complex_t) * PART_LEN1);
}

/*
 * 主处理接口：每次处理 PART_LEN=64 个采样
 *
 * 数据流：
 *   farEnd(64) ──┬── overlap-save 128点 ──► FFT ──► farend_fft
 *                └── 更新 xPow（远端平滑功率谱）
 *
 *   nearEnd(64) ─┬── overlap-save 128点 ──► FFT ──► nearend_fft
 *                └── 更新 dPow（近端平滑功率谱）→ dMinPow（舒适噪声估计）
 *
 *   EchoSubtraction(farend_fft, nearEnd, xPow, wfBuf) → e[64]
 *   EchoSuppression(nearend_overlap, farend_overlap, e) → output[64]
 */
void AECV2_Process(AECV2 *aecv2, float* farEnd, float* nearEnd,float* output){
  int i;

  float fft[PART_LEN2];
  float farend_overlap[PART_LEN2];
  float nearend_overlap[PART_LEN2];

  float farend_fft[2][PART_LEN1];
  float nearend_fft[2][PART_LEN1];

  float echo_subtractor_output[PART_LEN];

  float far_spectrum = 0.0f;
  float near_spectrum = 0.0f;

  const float gPow[2] = {0.9f, 0.1f};

  // Noise estimate constants.
  const int noiseInitBlocks = 500 * aecv2->mult;
  const float step = 0.1f;
  const float ramp = 1.0002f;
  const float gInitNoise[2] = {0.999f, 0.001f};

  // 远端信号FFT 128字节 FFT结果65长度复数
  memcpy(farend_overlap,aecv2->farend_old,sizeof(float)*PART_LEN);
  memcpy(farend_overlap+PART_LEN,farEnd,sizeof(float)*PART_LEN);
  memcpy(fft,farend_overlap,sizeof(float)*PART_LEN2);
  memcpy(aecv2->farend_old,farEnd,sizeof(float)*PART_LEN);
  Fft(fft, farend_fft);
  
  memcpy(nearend_overlap,aecv2->nearend_old,sizeof(float)*PART_LEN);
  memcpy(nearend_overlap+PART_LEN,nearEnd,sizeof(float)*PART_LEN);
  memcpy(fft,nearend_overlap,sizeof(float)*PART_LEN2);
  memcpy(aecv2->nearend_old,nearEnd,sizeof(float)*PART_LEN);
  Fft(fft, nearend_fft);

  for (i = 0; i < PART_LEN1; ++i) {
    //far_spectrum 远端频段能量
    //num_partitions 滤波器组�?12�?
    #if 1
    far_spectrum = farend_fft[0][i] * farend_fft[0][i] + farend_fft[1][i] * farend_fft[1][i];  //各个频段的实部的平方+虚部的平�?
    //为什么要*num_partitions 因为回声信号估算时，远端信号和滤波器系数是乘加关系，�?以这里的xPow也应该包�?12组远端信号的状�??
    aecv2->xPow[i] = gPow[0] * aecv2->xPow[i] + gPow[1] * aecv2->num_partitions * far_spectrum;  //xPow[i] = 0.9*xPow[i] + 0.1*12*far_spectrum
    // Calculate the magnitude spectrum.
    // 复数的模
    // abs_far_spectrum[i] = sqrtf(far_spectrum); //好像没有其他地方用到
    #else
    // 为了计算的更精确，需要将之前保存的远端信号能量一起放进来计算
    aecv2->far_spectrum_buf[i*kNormalNumPartitions + kNormalNumPartitions-1]  = farend_fft[0][i] * farend_fft[0][i] + farend_fft[1][i] * farend_fft[1][i];  //各个频段的实部的平方+虚部的平�?
    float far_spectrum_sum = 0;
    for(int j = 0; j < kNormalNumPartitions; j++){
      far_spectrum_sum += aecv2->far_spectrum_buf[i*kNormalNumPartitions + j];
    }
    aecv2->xPow[i] = gPow[0] * aecv2->xPow[i] + gPow[1] * far_spectrum_sum;
    memmove(aecv2->far_spectrum_buf+i*kNormalNumPartitions, aecv2->far_spectrum_buf+i*kNormalNumPartitions+1, (kNormalNumPartitions-1)*sizeof(float));
    #endif
  }
  
  //近端信号频段能量
  for (i = 0; i < PART_LEN1; ++i) {
    near_spectrum = nearend_fft[0][i] * nearend_fft[0][i] + nearend_fft[1][i] * nearend_fft[1][i];
    aecv2->dPow[i] = gPow[0] * aecv2->dPow[i] + gPow[1] * near_spectrum;
    // Calculate the magnitude spectrum.
    // abs_near_spectrum[i] = sqrtf(near_spectrum); //好像没有其他地方用到
  }

  // Estimate noise power. Wait until dPow is more stable.
  // dMinPow保存每个频段�?小的dPow�?
  // 舒�?�噪声相�?
  if (aecv2->noiseEstCtr > 50) {
    for (i = 0; i < PART_LEN1; i++) {
      if (aecv2->dPow[i] < aecv2->dMinPow[i]) {
        aecv2->dMinPow[i] = (aecv2->dPow[i] + step * (aecv2->dMinPow[i] - aecv2->dPow[i])) * ramp;   //step 0.1 ramp 1.0002
      } 
      else {
        aecv2->dMinPow[i] *= ramp;
      }
    }
  }

  // Smooth increasing noise power from zero at the start,
  // to avoid a sudden burst of comfort noise.
  // noiseInitBlocks 1000
  if (aecv2->noiseEstCtr < noiseInitBlocks) {
    aecv2->noiseEstCtr++;
    for (i = 0; i < PART_LEN1; i++) {
      //不能突然过大
      if (aecv2->dMinPow[i] > aecv2->dInitMinPow[i]) {
        aecv2->dInitMinPow[i] = gInitNoise[0] * aecv2->dInitMinPow[i] + gInitNoise[1] * aecv2->dMinPow[i];
      }
      else {
        aecv2->dInitMinPow[i] = aecv2->dMinPow[i];
      }
    }
    aecv2->noisePow = aecv2->dInitMinPow;
  }
  else {
    aecv2->noisePow = aecv2->dMinPow;
  }

  // Perform echo subtraction.
  // 标准的NLMS做回声消�?
  EchoSubtraction(aecv2->num_partitions,
                  &aecv2->extreme_filter_divergence,
                  aecv2->filter_step_size,
                  aecv2->error_threshold,
                  &farend_fft[0][0],
                  &aecv2->xfBufBlockPos,
                  aecv2->xfBuf,
                  nearEnd,
                  aecv2->xPow,
                  aecv2->wfBuf,
                  echo_subtractor_output);
  
  // Perform echo suppression.
  // nearend_extended_block_lowest_band 近端信号128字节
  // farend_extended_block_lowest_band 远端信号128字节
  // echo_subtractor_output 误差信号 64字节
  EchoSuppression(aecv2, nearend_overlap,
                  farend_overlap, echo_subtractor_output,
                  output);
}

AECV2 *AECV2_Create(void){
  int i;
  
  #if defined(ARCH_X86_FAMILY)
  use_sse2_ = (GetCPUInfo(kSSE2) != 0);
  #else
  use_sse2_ = false;
  #endif

  AECV2* aecv2 = (AECV2*)(malloc(sizeof(AECV2)));
  if(aecv2 == NULL){
    printf("aecv2 malloc failed\n");
    return NULL;
  }

  //�?查一下所有的数据有无初始�?
  aecv2->num_partitions = kNormalNumPartitions;

  memset(aecv2->farend_old,0,sizeof(aecv2->farend_old));
  memset(aecv2->nearend_old,0,sizeof(aecv2->nearend_old));

  memset(aecv2->xPow, 0, sizeof(aecv2->xPow));
  memset(aecv2->dPow, 0, sizeof(aecv2->dPow));
  memset(aecv2->dInitMinPow, 0, sizeof(aecv2->dInitMinPow));
  aecv2->noisePow = aecv2->dInitMinPow;
  aecv2->noiseEstCtr = 0;

  // Initial comfort noise power
  for (i = 0; i < PART_LEN1; i++) {
    aecv2->dMinPow[i] = 1.0e6f;
  }
  
  // Holds the last block written to
  aecv2->xfBufBlockPos = 0;
  // TODO(peah): Investigate need for these initializations. Deleting them
  // doesn't change the output at all and yields 0.4% overall speedup.
  memset(aecv2->xfBuf, 0, sizeof(complex_t) * kNormalNumPartitions * PART_LEN1);
  memset(aecv2->wfBuf, 0, sizeof(complex_t) * kNormalNumPartitions * PART_LEN1);
  memset(aecv2->coherence_state.sde, 0, sizeof(complex_t) * PART_LEN1);
  memset(aecv2->coherence_state.sxd, 0, sizeof(complex_t) * PART_LEN1);
  memset(aecv2->xfwBuf, 0, sizeof(complex_t) * kNormalNumPartitions * PART_LEN1);
  memset(aecv2->coherence_state.se, 0, sizeof(float) * PART_LEN1);

  memset(aecv2->eBuf, 0, sizeof(aecv2->eBuf));
  memset(aecv2->outBuf, 0, sizeof(float) * PART_LEN);

  // To prevent numerical instability in the first block.
  for (i = 0; i < PART_LEN1; i++) {
    aecv2->coherence_state.sd[i] = 1;
  }
  for (i = 0; i < PART_LEN1; i++) {
    aecv2->coherence_state.sx[i] = 1;
  }

  aecv2->extreme_filter_divergence = 0;
  aecv2->delayEstCtr = 0;
  aecv2->divergeState = 0;
  aecv2->delayIdx = 0;

  aecv2->hNlXdAvgMin = 1;
  aecv2->stNearState = 0;
  aecv2->echoState = 0;
  aecv2->hNlFbMin = 1;
  aecv2->hNlFbLocalMin = 1;
  aecv2->hNlNewMin = 0;
  aecv2->hNlMinCtr = 0;

  aecv2->overDrive = 2;
  aecv2->overdrive_scaling = 2;
  aecv2->nlp_mode = 1;
  aecv2->seed = 777;

  aecv2->sampFreq = 16000;
  aecv2->mult = 2;
  aecv2->filter_step_size = 0.5f;
  aecv2->error_threshold = 1.5e-6f;

  return aecv2;
}

void AECV2_Free(AECV2 *aecv2){
  if(aecv2 != NULL){
      free(aecv2);
      aecv2 = NULL;
  }
}

void AECV2_Delay_Set(AECV2 *aecv2,int delay){
  aecv2->delayIdx = delay;
}

void AECV2_Init(AECV2 *aecv2,int samplingFreq,int nlpMode){
  aecv2->mult = samplingFreq/8000;

  if (aecv2->sampFreq == 8000) {
    aecv2->filter_step_size = 0.6f;
  }
  else {
    aecv2->filter_step_size = 0.5f;
  }

  if (aecv2->sampFreq == 8000) {
    aecv2->error_threshold = 2e-6f;
  }
  else {
    aecv2->error_threshold = 1.5e-6f;
  }
  aecv2->delayIdx = 1;
}