/* Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"

#define SQUARE(x) ((x)*(x))

#define SMOOTH_BANDS 1


#if SMOOTH_BANDS
#ifdef SAMPLE48
#define NB_BANDS 22
#else
#define NB_BANDS 18
#endif // SAMPLE48
#else
#define NB_BANDS 21
#endif

#ifdef SAMPLE48
#define INPUT_SIZE 42
#else
#define INPUT_SIZE 38
#endif

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define MM_PI 3.14159265358979323846

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)

#define maxf(a,b) (((a) > (b)) ? (a):(b))
#define minf(a,b) (((a) < (b)) ? (a):(b))

#define NLPOPTION

#ifndef TRAINING
#define TRAINING 0
#endif
rnnoisecomputegru rnnoise_compute_gru;
rnnoisecomputedense rnnoise_compute_dense;

static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};


static const float weight[161] = {1.0f, 1.0187f, 1.0375f, 1.0563f, 1.075f, 1.0938f, 1.1125f, 1.1312f, 1.15f, 1.1688f, 1.1875f, 1.2062f, 
1.225f, 1.2438f, 1.2625f, 1.2812f, 1.3f, 1.3187f, 1.3375f, 1.3563f, 1.375f, 1.3937f, 1.4125f, 1.4313f, 1.45f, 1.4688f, 1.4875f, 1.5062f, 1.525f,
1.5438f, 1.5625f, 1.5812f, 1.6f, 1.6188f, 1.6375f, 1.6562f, 1.675f, 1.6937f, 1.7125f, 1.7313f, 1.75f, 1.7687f, 1.7875f, 1.8063f, 1.825f, 1.8438f,
1.8625f, 1.8812f, 1.9f, 1.9188f, 1.9375f, 1.9562f, 1.975f, 1.9938f, 2.0125f, 2.0312f, 2.05f, 2.0687f, 2.0875f, 2.1062f, 2.125f, 2.1438f, 2.1625f,
2.1813f, 2.2f, 2.2188f, 2.2375f, 2.2562f, 2.275f, 2.2937f, 2.3125f, 2.3313f, 2.35f, 2.3688f, 2.3875f, 2.4062f, 2.425f, 2.4437f, 2.4625f, 2.4812f,
2.5f, 2.5188f, 2.5375f, 2.5563f, 2.575f, 2.5938f, 2.6125f, 2.6312f, 2.65f, 2.6687f, 2.6875f, 2.7063f, 2.725f, 2.7438f, 2.7625f, 2.7812f, 2.8f,
2.8187f, 2.8375f, 2.8562f, 2.875f, 2.8938f, 2.9125f, 2.9313f, 2.95f, 2.9688f, 2.9875f, 3.0062f, 3.025f, 3.0437f, 3.0625f, 3.0813f, 3.1f, 3.1188f,
3.1375f, 3.1562f, 3.175f, 3.1937f, 3.2125f, 3.2312f, 3.25f, 3.2688f, 3.2875f, 3.3063f, 3.325f, 3.3438f, 3.3625f, 3.3812f, 3.4f, 3.4187f, 3.4375f,
3.4563f, 3.475f, 3.4938f, 3.5125f, 3.5312f, 3.55f, 3.5687f, 3.5875f, 3.6062f, 3.625f, 3.6438f, 3.6625f, 3.6813f, 3.7f, 3.7188f, 3.7375f, 3.7562f,
3.775f, 3.7937f, 3.8125f, 3.8313f, 3.85f, 3.8688f, 3.8875f, 3.9062f, 3.925f, 3.9437f, 3.9625f, 3.9812f, 4.0f};
typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE];
  float dct_table[NB_BANDS*NB_BANDS];
} CommonState;

struct DenoiseState {
  float analysis_mem[FRAME_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float pitch_enh_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
  int processcount;
  CommonState common;
  RNNState rnn;
};

#if SMOOTH_BANDS
void comput_spectrum_energy(float *sp_e, const kiss_fft_cpx *X)
{
	// int j = 0;
	// for (j = 0; j < FREQ_SIZE; j++)
	// {
	// 	sp_e[j] = SQUARE(X[j].r) +
	// 		SQUARE(X[j].i);
	// }
  int j = 0;
  while (j + 4 < FREQ_SIZE) {
    sp_e[j + 0] = SQUARE(X[j + 0].r) + SQUARE(X[j + 0].i);
    sp_e[j + 1] = SQUARE(X[j + 1].r) + SQUARE(X[j + 1].i);
    sp_e[j + 2] = SQUARE(X[j + 2].r) + SQUARE(X[j + 2].i);
    sp_e[j + 3] = SQUARE(X[j + 3].r) + SQUARE(X[j + 3].i);
    j += 4;
  }
	for (; j < FREQ_SIZE; j++) {
		sp_e[j] = SQUARE(X[j].r) + SQUARE(X[j].i);
	}  
}

void compute_x_band_energy(float *bandE, float *sp_e) {
	int i;
	float sum[NB_BANDS] = { 0 };
	for (i = 0; i<NB_BANDS - 1; i++)
	{
		int j;
		int band_size;
		band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
		// for (j = 0; j<band_size; j++) {
		// 	//float tmp;
		// 	float frac = (float)j / band_size;
		// 	sum[i] += (1 - frac)*sp_e[(eband5ms[i] << FRAME_SIZE_SHIFT) + j];
		// 	sum[i + 1] += frac*sp_e[(eband5ms[i] << FRAME_SIZE_SHIFT) + j];
		// }
    for (j = 0; j < band_size; j += 4) {
      float frac1 = (float)(j + 0) / band_size;
      float frac2 = (float)(j + 1) / band_size;
      float frac3 = (float)(j + 2) / band_size;
      float frac4 = (float)(j + 3) / band_size;
      float sp_e1 = sp_e[(eband5ms[i] << FRAME_SIZE_SHIFT) + j + 0];
      float sp_e2 = sp_e[(eband5ms[i] << FRAME_SIZE_SHIFT) + j + 1];
      float sp_e3 = sp_e[(eband5ms[i] << FRAME_SIZE_SHIFT) + j + 2];
      float sp_e4 = sp_e[(eband5ms[i] << FRAME_SIZE_SHIFT) + j + 3];

      sum[i] += (1 - frac1) * sp_e1;
      sum[i] += (1 - frac2) * sp_e2;
      sum[i] += (1 - frac3) * sp_e3;
      sum[i] += (1 - frac4) * sp_e4;
      sum[i + 1] += frac1 * sp_e1;
      sum[i + 1] += frac2 * sp_e2;
      sum[i + 1] += frac3 * sp_e3;
      sum[i + 1] += frac4 * sp_e4;
    }
	}
	sum[0] *= 2;
	sum[NB_BANDS - 1] *= 2;
	for (i = 0; i<NB_BANDS; i++)
	{
		bandE[i] = sum[i];
	}
}

void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    // for (j=0;j<band_size;j++) {
    //   float tmp;
    //   float frac = (float)j/band_size;
    //   tmp = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
    //   tmp += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
    //   sum[i] += (1-frac)*tmp;
    //   sum[i+1] += frac*tmp;
    // }
    for (j = 0; j < band_size; j += 4) {
      float tmp1;
      float tmp2;
      float tmp3;
      float tmp4;
      tmp1 = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 0].r);
      tmp2 = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 1].r);
      tmp3 = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 2].r);
      tmp4 = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 3].r);
      tmp1 += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 0].i);
      tmp2 += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 1].i);
      tmp3 += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 2].i);
      tmp4 += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 3].i);

      float frac1 = (float)(j + 0) / band_size;
      float frac2 = (float)(j + 1) / band_size;
      float frac3 = (float)(j + 2) / band_size;
      float frac4 = (float)(j + 3) / band_size;
      sum[i] += (1 - frac1) * tmp1;
      sum[i] += (1 - frac2) * tmp2;
      sum[i] += (1 - frac3) * tmp3;
      sum[i] += (1 - frac4) * tmp4;
      sum[i + 1] += frac1 * tmp1; 
      sum[i + 1] += frac2 * tmp2;
      sum[i + 1] += frac3 * tmp3;
      sum[i + 1] += frac4 * tmp4;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    // for (j=0;j<band_size;j++) {
    //   float tmp;
    //   float frac = (float)j/band_size;
    //   tmp = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r;
    //   tmp += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i;
    //   sum[i] += (1-frac)*tmp;
    //   sum[i+1] += frac*tmp;
    // }
      for (j = 0; j < band_size; j += 4) {
      float tmp1;
      float tmp2;
      float tmp3;
      float tmp4;
      tmp1 = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 0].r * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 0].r;
      tmp2 = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 1].r * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 1].r;
      tmp3 = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 2].r * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 2].r;
      tmp4 = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 3].r * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 3].r;
      tmp1 += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 0].i * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 0].i;
      tmp2 += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 1].i * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 1].i;
      tmp3 += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 2].i * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 2].i;
      tmp4 += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 3].i * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 3].i;

      float frac1 = (float)(j + 0) / band_size;
      float frac2 = (float)(j + 1) / band_size;
      float frac3 = (float)(j + 2) / band_size;
      float frac4 = (float)(j + 3) / band_size;

      sum[i] += (1 - frac1) * tmp1;
      sum[i] += (1 - frac2) * tmp2;
      sum[i] += (1 - frac3) * tmp3;
      sum[i] += (1 - frac4) * tmp4;
      sum[i+1] += frac1 * tmp1;
      sum[i+1] += frac2 * tmp2;
      sum[i+1] += frac3 * tmp3;
      sum[i+1] += frac4 * tmp4;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void interp_band_gain(float *g, /*const*/ float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    // for (j=0;j<band_size;j++) {
    //   float frac = (float)j/band_size;
    //   g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    // }
    for (j = 0; j < band_size; j += 4) {
      float frac1 = (float)(j + 0) / band_size;
      float frac2 = (float)(j + 1) / band_size;
      float frac3 = (float)(j + 2) / band_size;
      float frac4 = (float)(j + 3) / band_size;
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 0] = (1 - frac1) * bandE[i] + frac1 * bandE[i+1];
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 1] = (1 - frac2) * bandE[i] + frac2 * bandE[i+1];
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 2] = (1 - frac3) * bandE[i] + frac3 * bandE[i+1];
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j + 3] = (1 - frac4) * bandE[i] + frac4 * bandE[i+1];
    }
  }
	//int i;
	//memset(g, 0, FREQ_SIZE);
	//for (i = 0; i<NB_BANDS-1; i++)
	//{
	//	int j;
	//	for (j = 0; j<(eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT; j++)
	//		g[(eband5ms[i] << FRAME_SIZE_SHIFT) + j] = bandE[i];
	//}
}
#else
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    opus_val32 sum = 0;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++) {
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
    }
    bandE[i] = sum;
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++)
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = bandE[i];
  }
}
#endif




static void check_init(DenoiseState *st) {
  int i;
  if (st->common.init) return;
  st->common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE;i++)
    st->common.half_window[i] = sin(.5*MM_PI*sin(.5*MM_PI*(i+.5)/FRAME_SIZE) * sin(.5*MM_PI*(i+.5)/FRAME_SIZE));
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      st->common.dct_table[i*NB_BANDS + j] = cos((i+.5)*j*MM_PI/NB_BANDS);
      if (j==0) st->common.dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }
  st->common.init = 1;
}

static void dct(DenoiseState *st,float *out, const float *in) {
  int i;
  check_init(st);
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * st->common.dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./22);
  }
}
#endif

static void forward_transform(DenoiseState *st,kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init(st);
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(st->common.kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

static void inverse_transform(DenoiseState *st,float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init(st);
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(st->common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

static void apply_window(DenoiseState *st,float *x) {
  int i;
  check_init(st);
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= st->common.half_window[i];
    x[WINDOW_SIZE - 1 - i] *= st->common.half_window[i];
  }
}

int rnnoise_get_size(void) {
  return sizeof(DenoiseState);
}

int rnnoise_init(DenoiseState *st) {
  memset(st, 0, sizeof(*st));
  rnnoise_init_rnn();
  return 0;
}

void *rnnoise_create(void) {
  DenoiseState *st;
  st = malloc(rnnoise_get_size());
  init_pitch_sse2();
  rnnoise_init(st);
  return (void *)st;
}

void rnnoise_destroy(void *stt) {
  DenoiseState *st = NULL;
  st = (DenoiseState *)stt;
  if (st->common.kfft != NULL)
  {
	  opus_fft_free(st->common.kfft, 0);
  }
  free(st);
}

#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex,float *SP_E, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(st,x);
  forward_transform(st,X, x);
#if TRAINING
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
#endif
  comput_spectrum_energy(SP_E,X);
  compute_x_band_energy(Ex,SP_E);
  //compute_band_energy(Ex, X);
}

static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
	float *Ex, float *Ep, float *Exp, float *features, float *SP_E, const float *in) {
  int i;
  float E = 0;
  float *ceps_0, *ceps_1, *ceps_2;
  float spec_variability = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE>>1];
  int pitch_index;
  float gain;
  float *(pre[1]);
  float tmp[NB_BANDS];
  float follow, logMax;
  frame_analysis(st, X, Ex, SP_E, in);
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);
  pre[0] = &st->pitch_buf[0];
  pitch_downsample_rnn(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  pitch_search_rnn(pitch_buf+(PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
               PITCH_XCORR, &pitch_index);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = remove_doubling_rnn(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  for (i=0;i<WINDOW_SIZE;i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];
  apply_window(st,p);
  forward_transform(st,P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i]/sqrt(.001+Ex[i]*Ep[i]);
  dct(st,tmp, Exp);
  for (i=0;i<NB_DELTA_CEPS;i++) features[NB_BANDS+2*NB_DELTA_CEPS+i] = tmp[i];
  features[NB_BANDS+2*NB_DELTA_CEPS] -= 1.3f;
  features[NB_BANDS+2*NB_DELTA_CEPS+1] -= 0.9f;
  features[NB_BANDS+3*NB_DELTA_CEPS] = .01*(pitch_index-300);
  logMax = -2;
  follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-1.5, Ly[i]);
    E += Ex[i];
  }
  if (!TRAINING && E < 0.04) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
  }
  dct(st,features, Ly);
  features[0] -= 12;
  features[1] -= 4;
  ceps_0 = st->cepstral_mem[st->memid];
  ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM+st->memid-1] : st->cepstral_mem[st->memid-1];
  ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM+st->memid-2] : st->cepstral_mem[st->memid-2];
  for (i=0;i<NB_BANDS;i++) ceps_0[i] = features[i];
  st->memid++;
  for (i=0;i<NB_DELTA_CEPS;i++) {
    features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
    features[NB_BANDS+i] = ceps_0[i] - ceps_2[i];
    features[NB_BANDS+NB_DELTA_CEPS+i] =  ceps_0[i] - 2*ceps_1[i] + ceps_2[i];
  }
  /* Spectral variability features. */
  if (st->memid == CEPS_MEM) st->memid = 0;
  for (i=0;i<CEPS_MEM;i++)
  {
    int j;
    float mindist = 1e15f;
    for (j=0;j<CEPS_MEM;j++)
    {
      int k;
      float dist=0;
      for (k=0;k<NB_BANDS;k++)
      {
        float tmp;
        tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
        dist += tmp*tmp;
      }
      if (j!=i)
        mindist = MIN32(mindist, dist);
    }
    spec_variability += mindist;
  }
  features[NB_BANDS+3*NB_DELTA_CEPS+1] = spec_variability/CEPS_MEM-2.1;
  return TRAINING && E < 0.1;
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(st,x, y);
  apply_window(st,x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  //kiss_fft_cpx X1[FREQ_SIZE];

  for (i=0;i<NB_BANDS;i++) {
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
	if (r[i] >= 1)
		r[i] = 1;
  }
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r += rf[i]*P[i].r;
    X[i].i += rf[i]*P[i].i;
  }
  float newE[NB_BANDS];
  compute_band_energy(newE, X);
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
}

void compute_rnn(RNNState *rnn, float *gains, float *vad, const float *input, int procCount) {
	int i;
	float dense_out[MAX_NEURONS];
	float noise_input[MAX_NEURONS * 3];
	float denoise_input[MAX_NEURONS * 3];
	rnnoise_compute_dense(&input_dense, dense_out, input);
	rnnoise_compute_gru(&vad_gru, rnn->vad_gru_state, dense_out);
	//if (procCount != 20)
		rnnoise_compute_dense(&vad_output, vad, rnn->vad_gru_state);
	for (i = 0; i<INPUT_DENSE_SIZE; i++) noise_input[i] = dense_out[i];
	for (i = 0; i<VAD_GRU_SIZE; i++) noise_input[i + INPUT_DENSE_SIZE] = rnn->vad_gru_state[i];
	for (i = 0; i<INPUT_SIZE; i++) noise_input[i + INPUT_DENSE_SIZE + VAD_GRU_SIZE] = input[i];
	rnnoise_compute_gru(&noise_gru, rnn->noise_gru_state, noise_input);

	for (i = 0; i<VAD_GRU_SIZE; i++) denoise_input[i] = rnn->vad_gru_state[i];
	for (i = 0; i<NOISE_GRU_SIZE; i++) denoise_input[i + VAD_GRU_SIZE] = rnn->noise_gru_state[i];
	for (i = 0; i<INPUT_SIZE; i++) denoise_input[i + VAD_GRU_SIZE + NOISE_GRU_SIZE] = input[i];
	rnnoise_compute_gru(&denoise_gru, rnn->denoise_gru_state, denoise_input);
	rnnoise_compute_dense(&denoise_output, gains, rnn->denoise_gru_state);
}

float rnnoise_process_frame(void *stt, float *out, const float *in,float *low_band_averge_g) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[WINDOW_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS] = {0};
  float gf[FREQ_SIZE]={1};
  float sp_e[FREQ_SIZE], sp_e1[FREQ_SIZE],sp_e_in[FREQ_SIZE],sp_e_out[FREQ_SIZE];
  float averge_g = 0.0f;
  float vad_prob = 0.0f;
  float vad_cal = 0.0f;
  int silence = 0;
  static float smdb = 26.0;
  float db = 0.0;
  float override = 0.0;
  float maxgg = 0.0, maxg = 0.0/*, lowg = 1.0*/;
  //float lowsumspe = 0.0, lowdb = 0.0;
  //static float smlowdb = 5;
  float sumspe = 0.0, sumspe1 = 0.0;
  //static unsigned int count = 0;
  int itemp = 0;
  float lavg = 0.0, havg = 0.0,hl = 0.0;
  static float sspe = 38, sspe1 = 12000;
  static const float a_hp[2] = {-1.99599f, 0.99600f};
  static const float b_hp[2] = {-2.0f, 1.0f};
  DenoiseState *st = NULL;
  st = (DenoiseState *)stt;
  //count++;

  biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = compute_frame_features(st, X, P, Ex, Ep, Exp, features, sp_e, in);
  comput_spectrum_energy(sp_e_in, X);
  if (!silence) {
    compute_rnn(&st->rnn, g, &vad_prob, features, st->processcount);
    pitch_filter(X, P, Ex, Ep, Exp, g);
    for (i=0;i<NB_BANDS;i++) {
		float alpha = .6f;
      g[i] = MAX16(g[i], alpha*st->lastg[i]);
      st->lastg[i] = g[i];
    }

	//to overdrive low band addrate
#ifdef NLPOPTION
	for (i = 0; i < 5; i++)
	{
		maxg = maxf(maxg, g[i]);
	}

	for (i = 0; i < 5; i++) {
		float tmp = g[i] / maxg;
		g[i] *= tmp;
	}
#endif

    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      X[i].r *= gf[i];
      X[i].i *= gf[i];
    }
	comput_spectrum_energy(sp_e1, X);
#ifdef NLPOPTION
	for (i = 0; i < 40; i++){
		sumspe += sp_e[i];
		lavg += sp_e1[i];
	}

	for (i = 40; i < 100; i++) {
		sumspe += sp_e[i];
		sumspe1 += sp_e1[i];
	}

	for (i = 100; i < 140; i++) {
		sumspe += sp_e[i];
		havg += sp_e1[i];
	}
	for (i = 140; i < FREQ_SIZE; i++) {
		sumspe += sp_e[i];
		sumspe1 += sp_e1[i];
	}
	sumspe1 = lavg + havg;
	sumspe /= FREQ_SIZE;
	sumspe1 /= FREQ_SIZE;

	if (vad_prob < 0.1&& sumspe1 < 50&& lavg > havg)
	{
		sspe = sspe*0.98 + 0.02*sumspe;
	}
	else if (vad_prob >= 0.1)
	{
		sspe1 = sspe1*0.95 + 0.05*sumspe1;
	}

	db =10*log10f(sspe1 / sspe);
	hl = log10f(havg / lavg);
	db = db < 0 ? 0 : db;
	if (hl > 0.4) {
		db = 30;
		smdb = 0.7*smdb + 0.3*db;
	}
	else
	{
		smdb = 0.95*smdb + 0.05*db;
	}


	if (sspe > 40 && smdb < 25)
	{  
		if (smdb <= 5)
			override = 160;
		else 
			override =160 * (25 - smdb)/25;
		
		float gg[FREQ_SIZE] = {0.0};

		for (i = 0; i < FREQ_SIZE; i++)
		{
			gg[i] = sp_e1[i] / sp_e[i];
			maxgg = maxf(maxgg, gg[i]);
		}
		maxgg = minf(maxgg, 1);
		for (i = 0; i < FREQ_SIZE; i++)
		{
			itemp = (override - override * gg[i] / maxgg);
			if (itemp < 0)
				itemp = 0;
			X[i].r /= weight[itemp];
			X[i].i /= weight[itemp];
		}
	}
#endif
	if (st->processcount < 20)
	{
		st->processcount++;
		if (vad_prob < 0.7) vad_cal = 0.05 + st->processcount*st->processcount*0.00175; //reduce 26dB ~ 3dB
		else {
			st->processcount = 20;
			vad_cal = 1;
		}
		for (i = 0; i < FREQ_SIZE; i++)
		{
			X[i].r *= vad_cal;
			X[i].i *= vad_cal;
		}
	}
#endif
  }
  comput_spectrum_energy(sp_e_out, X);
    for (i = 0; i < FREQ_SIZE; i++)
  {
    float tmp = sp_e_in[i] / sp_e_out[i];
    if (tmp < 1.0)
    {
      X[i].r *= tmp;
      X[i].i *= tmp;
      if (i > 40) {
        averge_g += 1;
      }
    }
    else
    {
      if (i > 40) {
        averge_g += 1/tmp;
      }
    }
  }
  *low_band_averge_g = averge_g / 120;

  frame_synthesis(st, out, X);
  //for (i = 0; i < FRAME_SIZE; i++) out[i] = (out[i] > 32767?32767:(out[i] < -32768?-32768:out[i]));
  return vad_prob;
}

#if TRAINING

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

int main(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_hp_n[2]={0};
  float mem_resp_x[2]={0};
  float mem_resp_n[2]={0};
  float x[FRAME_SIZE];
  float n[FRAME_SIZE];
  float xn[FRAME_SIZE];
  int vad_cnt=0;
  int gain_change_count=0;
  float speech_gain = 1, noise_gain = 1;
  FILE *f1, *f2, *fout;
  DenoiseState *st;
  DenoiseState *noise_state;
  DenoiseState *noisy;
  st = rnnoise_create();
  noise_state = rnnoise_create();
  noisy = rnnoise_create();
  if (argc!=4) {
    fprintf(stderr, "usage: %s <speech> <noise> <output denoised>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "r");
  f2 = fopen(argv[2], "r");
  fout = fopen(argv[3], "w");
  for(i=0;i<150;i++) {
    short tmp[FRAME_SIZE];
    fread(tmp, sizeof(short), FRAME_SIZE, f2);
  }
  while (1) {
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[WINDOW_SIZE];
    float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS];
	float sp_e[FREQ_SIZE];
    float Exp[NB_BANDS];
    float Ln[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    float gf[FREQ_SIZE]={1};
    short tmp[FRAME_SIZE];
    float vad=0;
    float vad_prob;
    float E=0;
    if (count==50000000) break;
    if (++gain_change_count > 2821) {
      speech_gain = pow(10., (-40+(rand()%60))/20.);
      noise_gain = pow(10., (-30+(rand()%50))/20.);
      if (rand()%10==0) noise_gain = 0;
      noise_gain *= speech_gain;
      if (rand()%10==0) speech_gain = 0;
      gain_change_count = 0;
      rand_resp(a_noise, b_noise);
      rand_resp(a_sig, b_sig);
      lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
      for (i=0;i<NB_BANDS;i++) {
        if (eband5ms[i]<<FRAME_SIZE_SHIFT > lowpass) {
          band_lp = i;
          break;
        }
      }
    }
    if (speech_gain != 0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
      if (feof(f1)) {
        rewind(f1);
        fread(tmp, sizeof(short), FRAME_SIZE, f1);
      }
      for (i=0;i<FRAME_SIZE;i++) x[i] = speech_gain*tmp[i];
      for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) x[i] = 0;
      E = 0;
    }
    if (noise_gain!=0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f2);
      if (feof(f2)) {
        rewind(f2);
        fread(tmp, sizeof(short), FRAME_SIZE, f2);
      }
      for (i=0;i<FRAME_SIZE;i++) n[i] = noise_gain*tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) n[i] = 0;
    }
    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
    biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
    for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];
    if (E > 1e9f) {
      vad_cnt=0;
    } else if (E > 1e8f) {
      vad_cnt -= 5;
    } else if (E > 1e7f) {
      vad_cnt++;
    } else {
      vad_cnt+=2;
    }
    if (vad_cnt < 0) vad_cnt = 0;
    if (vad_cnt > 15) vad_cnt = 15;

    if (vad_cnt >= 10) vad = 0;
    else if (vad_cnt > 0) vad = 0.5f;
    else vad = 1.f;

    frame_analysis(st, Y, Ey, sp_e, x);
    frame_analysis(noise_state, N, En, sp_e, n);
    for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En[i]);
    int silence = compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, sp_e, xn);
    pitch_filter(X, P, Ex, Ep, Exp, g);
    //printf("%f %d\n", noisy->last_gain, noisy->last_period);
    for (i=0;i<NB_BANDS;i++) {
      g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
      if (g[i] > 1) g[i] = 1;
      if (silence || i > band_lp) g[i] = -1;
      if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
      if (vad==0 && noise_gain==0) g[i] = -1;
    }
    count++;
#if 0
    for (i=0;i<NB_FEATURES;i++) printf("%f ", features[i]);
    for (i=0;i<NB_BANDS;i++) printf("%f ", g[i]);
    for (i=0;i<NB_BANDS;i++) printf("%f ", Ln[i]);
    printf("%f\n", vad);
#endif
#if 1
    fwrite(features, sizeof(float), NB_FEATURES, stdout);
    fwrite(g, sizeof(float), NB_BANDS, stdout);
    fwrite(Ln, sizeof(float), NB_BANDS, stdout);
    fwrite(&vad, sizeof(float), 1, stdout);
#endif
#if 0
    compute_rnn(&noisy->rnn, g, &vad_prob, features);
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      X[i].r *= gf[i];
      X[i].i *= gf[i];
    }
#endif
    frame_synthesis(noisy, xn, X);

    for (i=0;i<FRAME_SIZE;i++) tmp[i] = xn[i];
    fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
#endif
  }
  fprintf(stderr, "matrix size: %d x %d\n", count, NB_FEATURES + 2*NB_BANDS + 1);
  fclose(f1);
  fclose(f2);
  fclose(fout);
  return 0;
}

#endif
