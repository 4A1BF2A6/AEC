#ifndef __AEC_V2_H__
#define __AEC_V2_H__

#include <stdbool.h>
#include <stdint.h>

#define PART_LEN 64               // Length of partition
#define PART_LEN1 (PART_LEN + 1)  // Unique fft coefficients
#define PART_LEN2 (PART_LEN * 2)  // Length of partition * 2

#define SPL_WORD16_MAX 32767
#define SPL_WORD16_MIN -32768

#define SPL_MAX(A, B) (A > B ? A : B)  // Get max value
#define SPL_MIN(A, B) (A < B ? A : B)  // Get min value
#define SPL_SAT(a, b, c) (b > a ? a : b < c ? c : b)

#define kNormalNumPartitions 12

extern bool use_sse2_;
extern const float kNormalSmoothingCoefficients[2][2];
extern const float kMinFarendPSD;

typedef float complex_t[2];

typedef struct CoherenceState {
  complex_t sde[PART_LEN1];  // cross-psd of nearend and error
  complex_t sxd[PART_LEN1];  // cross-psd of farend and nearend
  float sx[PART_LEN1], sd[PART_LEN1], se[PART_LEN1];  // far, near, error psd
} CoherenceState;

typedef struct {
  float nearend_old[PART_LEN];
  float farend_old[PART_LEN];

  int sampFreq;
  float filter_step_size;  // stepsize
  float error_threshold;   // error threshold

  float xPow[PART_LEN1];
  float dPow[PART_LEN1];
  float dMinPow[PART_LEN1];
  float dInitMinPow[PART_LEN1];
  float* noisePow;
  int noiseEstCtr;

  int num_partitions;
  int mult;  // sampling frequency multiple
  
  // Flag that extreme filter divergence has been detected by the Echo
  // Suppressor.
  int extreme_filter_divergence;

  int xfBufBlockPos;

  CoherenceState coherence_state;

  float xfBuf[2][kNormalNumPartitions * PART_LEN1];   // farend fft buffer
  float wfBuf[2][kNormalNumPartitions * PART_LEN1];   // filter fft buffer
  complex_t xfwBuf[kNormalNumPartitions * PART_LEN1]; // Farend windowed fft buffer.

  float far_spectrum_buf[kNormalNumPartitions * PART_LEN1];

  float eBuf[PART_LEN2];  // error
  float outBuf[PART_LEN];

  int delayEstCtr;
  short divergeState;
  int delayIdx;

  float hNlXdAvgMin;
  short stNearState, echoState;
  float overDrive;
  float overdrive_scaling;
  float hNlFbMin, hNlFbLocalMin;
  int hNlNewMin, hNlMinCtr;

  int nlp_mode;
  uint32_t seed;
}AECV2;

AECV2 *AECV2_Create();

void AECV2_Free(AECV2 *aecv2);

void AECV2_Delay_Set(AECV2 *aecv2,int delay);

void AECV2_Init(AECV2 *aecv2,int samplingFreq,int nlpMode);

void AECV2_Process(AECV2 *aecv2, float* farEnd, float* nearEnd,float* output);

#if defined(ARCH_X86_FAMILY)
void FilterFar_SSE2(int num_partitions,
                      int x_fft_buf_block_pos,
                      float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float y_fft[2][PART_LEN1]);

void ScaleErrorSignal_SSE2(float mu,
                          float error_threshold,
                          float x_pow[PART_LEN1],
                          float ef[2][PART_LEN1]);

void FilterAdaptation_SSE2(int num_partitions,
                          int x_fft_buf_block_pos,
                          float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                          float e_fft[2][PART_LEN1],
                          float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);

int PartitionDelay_SSE2(int num_partitions,
                        float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);

void UpdateCoherenceSpectra_SSE2(int mult,
                                float efw[2][PART_LEN1],
                                float dfw[2][PART_LEN1],
                                float xfw[2][PART_LEN1],
                                CoherenceState* coherence_state,
                                short* filter_divergence_state,
                                int* extreme_filter_divergence);

void ComputeCoherence_SSE2(const CoherenceState* coherence_state,
                          float* cohde,
                          float* cohxd);

void Overdrive_SSE2(float overdrive_scaling,
                    float hNlFb,
                    float hNl[PART_LEN1]);

void Suppress_SSE2(const float hNl[PART_LEN1], float efw[2][PART_LEN1]);
#endif

#if defined(MIPS_FPU_LE)
void FilterFar_MIPS(int num_partitions,
                      int x_fft_buf_block_pos,
                      float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float y_fft[2][PART_LEN1]);

void ScaleErrorSignal_MIPS(float mu,
                           float error_threshold,
                           float x_pow[PART_LEN1],
                           float ef[2][PART_LEN1]);

void FilterAdaptation_MIPS(int num_partitions,
                          int x_fft_buf_block_pos,
                          float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                          float e_fft[2][PART_LEN1],
                          float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);

void Overdrive_MIPS(float overdrive_scaling,
                    float hNlFb,
                    float hNl[PART_LEN1]);

void Suppress_MIPS(const float hNl[PART_LEN1],
                   float efw[2][PART_LEN1]);                      
#endif

#if defined(ARCH_ARM_NEON)
void FilterFar_NEON(int num_partitions,
                      int x_fft_buf_block_pos,
                      float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float h_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                      float y_fft[2][PART_LEN1]);

void ScaleErrorSignal_NEON(float mu,
                          float error_threshold,
                          float x_pow[PART_LEN1],
                          float ef[2][PART_LEN1]);

void FilterAdaptation_NEON(int num_partitions,
                          int x_fft_buf_block_pos,
                          float x_fft_buf[2][kNormalNumPartitions * PART_LEN1],
                          float e_fft[2][PART_LEN1],
                          float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);

int PartitionDelay_NEON(int num_partitions,
                        float h_fft_buf[2][kNormalNumPartitions * PART_LEN1]);

void UpdateCoherenceSpectra_NEON(int mult,
                                float efw[2][PART_LEN1],
                                float dfw[2][PART_LEN1],
                                float xfw[2][PART_LEN1],
                                CoherenceState* coherence_state,
                                short* filter_divergence_state,
                                int* extreme_filter_divergence);

void ComputeCoherence_NEON(const CoherenceState* coherence_state,
                          float* cohde,
                          float* cohxd);

void Overdrive_NEON(float overdrive_scaling,
                    float hNlFb,
                    float hNl[PART_LEN1]);

void Suppress_NEON(const float hNl[PART_LEN1], float efw[2][PART_LEN1]);

#endif

#ifdef _MSC_VER
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#else
#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))
#endif

extern ALIGN16_BEG const float ALIGN16_END weightCurve[65];
extern ALIGN16_BEG const float ALIGN16_END overDriveCurve[65];

#endif