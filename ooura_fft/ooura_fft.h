#ifndef __OOURA_FFT_H__
#define __OOURA_FFT_H__

#if defined(ARCH_X86_FAMILY)
void cft1st_128_SSE2(float* a);
void cftmdl_128_SSE2(float* a);
void rftfsub_128_SSE2(float* a);
void rftbsub_128_SSE2(float* a);
#endif

#if defined(MIPS_FPU_LE)
void cft1st_128_mips(float* a);
void cftmdl_128_mips(float* a);
void rftfsub_128_mips(float* a);
void rftbsub_128_mips(float* a);
#endif

#if defined(ARCH_ARM_NEON)
void cft1st_128_neon(float* a);
void cftmdl_128_neon(float* a);
void rftfsub_128_neon(float* a);
void rftbsub_128_neon(float* a);
#endif

extern const float rdft_w[64];
extern const float rdft_wk3ri_first[16];
extern const float rdft_wk3ri_second[16];



void OouraFft_Fft(float* a);
void OouraFft_InverseFft(float* a);

#endif  // __OOURA_FFT_H__
