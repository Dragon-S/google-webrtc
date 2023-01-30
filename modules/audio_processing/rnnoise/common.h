

#ifndef COMMON_H
#define COMMON_H

#include "stdlib.h"
#include "string.h"
#include "rnn.h"
#include "arch.h"
//#include "webrtc/system_wrappers/include/cpu_features_wrapper.h"

#define RNN_INLINE inline
#define OPUS_INLINE inline

//#define SAMPLE48

#define FRAME_SIZE_SHIFT 2
#ifdef SAMPLE48
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#else
#define FRAME_SIZE (40<<FRAME_SIZE_SHIFT)
#endif
#define WINDOW_SIZE (2*FRAME_SIZE)

#define FREQ_SIZE (FRAME_SIZE + 1)

#ifdef SAMPLE48
#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#else
#define PITCH_MIN_PERIOD 20
#define PITCH_MAX_PERIOD 256
#define PITCH_FRAME_SIZE 320
#endif
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)
#define PITCH_XCORR (PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD)

/** RNNoise wrapper for malloc(). To do your own dynamic allocation, all you need t
o do is replace this function and rnnoise_free */
#ifndef OVERRIDE_RNNOISE_ALLOC
static RNN_INLINE void *rnnoise_alloc (size_t size)
{
   return malloc(size);
}
#endif

/** RNNoise wrapper for free(). To do your own dynamic allocation, all you need to do is replace this function and rnnoise_alloc */
#ifndef OVERRIDE_RNNOISE_FREE
static RNN_INLINE void rnnoise_free (void *ptr)
{
   free(ptr);
}
#endif

/** Copy n elements from src to dst. The 0* term provides compile-time type checking  */
#ifndef OVERRIDE_RNN_COPY
#define RNN_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Copy n elements from src to dst, allowing overlapping regions. The 0* term
    provides compile-time type checking */
#ifndef OVERRIDE_RNN_MOVE
#define RNN_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Set n elements of dst to zero */
#ifndef OVERRIDE_RNN_CLEAR
#define RNN_CLEAR(dst, n) (memset((dst), 0, (n)*sizeof(*(dst))))
#endif

typedef void (*rnnoisecomputegru)(const GRULayer *gru, float *state, const float *input);
extern rnnoisecomputegru rnnoise_compute_gru;

typedef void (*rnnoisecomputedense)(const DenseLayer *layer, float *output, const float *input);
extern rnnoisecomputedense rnnoise_compute_dense;

typedef  void (*pitchxcorrkernel)(const float * x, const float * y, float sum[4], int len);
extern pitchxcorrkernel pitch_xcorr_kernel;

typedef void(*pitchdualinnerprod)(const float *x, const float *y01, const float *y02,
	int N, float *xy1, float *xy2);
extern pitchdualinnerprod pitch_dual_inner_prod;

typedef float(*pitchceltinnerprod)(const float *x, const float *y, int N);
extern pitchceltinnerprod pitch_celt_inner_prod;
#endif
