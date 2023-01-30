/* Copyright (c) 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
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

#include <math.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "tansig_table.h"
#include "rnn.h"
#include "rnn_data.h"
#include <stdio.h>

static OPUS_INLINE float tansig_approx(float x)
{
    int i;
    float y;//, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
       return 0;
#endif
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    // dy = 1-y*y;
    // y = y + x*dy*(1 - y*x);
    y = y + x * tansig_diff_table[i];
    return sign*y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
  return .5 + .5*tansig_approx(.5*x);
}

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}

static void compute_dense(const DenseLayer *layer, float *output, const float *input)
{
   int i, j;
   int N, M,im;
   //int stride;
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   //stride = N;

   for (i=0;i<N;i++)
   {
      /* Compute update gate. */
	  im = i * M;
      float sum = layer->bias[i];

    //   for (j=0;j<M;j++)
    //     sum += layer->input_weights[im + j]* input[j];
      j = 0;
      float sum1 = 0.0f;
      float sum2 = 0.0f;
      float sum3 = 0.0f;
      float sum4 = 0.0f;
      while(j + 4 < M) {
          sum1 += layer->input_weights[im + j + 0] * input[j + 0];
          sum2 += layer->input_weights[im + j + 1] * input[j + 1];
          sum3 += layer->input_weights[im + j + 2] * input[j + 2];
          sum4 += layer->input_weights[im + j + 3] * input[j + 3];
          j += 4;
      }
      for (; j < M; j++) {
          sum += layer->input_weights[im + j] * input[j];
      }
      sum += sum1 + sum2 + sum3 + sum4;
      
      output[i] = WEIGHTS_SCALE*sum;
   }
   if (layer->activation == ACTIVATION_SIGMOID) {
      for (i=0;i<N;i++)
         output[i] = sigmoid_approx(output[i]);
   } else if (layer->activation == ACTIVATION_TANH) {
      for (i=0;i<N;i++)
         output[i] = tansig_approx(output[i]);
   } else if (layer->activation == ACTIVATION_RELU) {
      for (i=0;i<N;i++)
         output[i] = relu(output[i]);
   } else {
     //*(int*)0=0;
   }
}

static void compute_gru(const GRULayer *gru, float *state, const float *input)
{
	int i, j;
	int N, M,im,in;
	//int stride;
	float z[MAX_NEURONS];
	float r[MAX_NEURONS];
	float h[MAX_NEURONS];
	M = gru->nb_inputs;
	N = gru->nb_neurons;
	//stride = 3 * N;
	for (i = 0; i < N; i++)
	{
		/* Compute update gate. */
		im = i * M;
		in = i * N;
		float sum = gru->bias[i];
		// for (j = 0; j < M; j++)
		// 	sum += gru->update_input_weights[im + j] * input[j];
        j = 0;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sum4 = 0.0f;
        while (j + 4 < M) {
            sum1 += gru->update_input_weights[im + j + 0] * input[j + 0];
            sum2 += gru->update_input_weights[im + j + 1] * input[j + 1];
            sum3 += gru->update_input_weights[im + j + 2] * input[j + 2];
            sum4 += gru->update_input_weights[im + j + 3] * input[j + 3];
            j += 4;
        }
        for (; j < M; j++) {
            sum += gru->update_input_weights[im + j] * input[j];
        }

		// for (j = 0; j < N; j++)
		// 	sum += gru->update_recurrent_weights[in + j] * state[j];
        j = 0;
        while (j + 4 < N) {
            sum1 += gru->update_recurrent_weights[in + j + 0] * state[j + 0];
            sum2 += gru->update_recurrent_weights[in + j + 1] * state[j + 1];
            sum3 += gru->update_recurrent_weights[in + j + 2] * state[j + 2];
            sum4 += gru->update_recurrent_weights[in + j + 3] * state[j + 3];
            j += 4;
        }
        for (; j < N; j++) {
            sum += gru->update_recurrent_weights[in + j] * state[j];
        }
        sum += sum1 + sum2 + sum3 + sum4;

		z[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
	}
	for (i = 0; i<N; i++)
	{
		/* Compute reset gate. */
		im = i * M;
		in = i * N;
		float sum = gru->bias[N + i];
		// for (j = 0; j<M; j++)
		// 	sum += gru->reset_input_weights[im + j] * input[j];
        j = 0;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sum4 = 0.0f;
        while (j + 4 < M) {
            sum1 += gru->reset_input_weights[im + j + 0] * input[j + 0];
            sum2 += gru->reset_input_weights[im + j + 1] * input[j + 1];
            sum3 += gru->reset_input_weights[im + j + 2] * input[j + 2];
            sum4 += gru->reset_input_weights[im + j + 3] * input[j + 3];
            j += 4;
        }
        for (; j < M; j++) {
            sum += gru->reset_input_weights[im + j] * input[j];
        }

		// for (j = 0; j<N; j++)
		// 	sum += gru->reset_recurrent_weights[in + j] * state[j];
        j = 0;
        while (j + 4 < N) {
            sum1 += gru->reset_recurrent_weights[in + j + 0] * state[j + 0];
            sum2 += gru->reset_recurrent_weights[in + j + 1] * state[j + 1];
            sum3 += gru->reset_recurrent_weights[in + j + 2] * state[j + 2];
            sum4 += gru->reset_recurrent_weights[in + j + 3] * state[j + 3];
            j += 4;
        }
        for (; j < N; j++) {
            sum += gru->reset_recurrent_weights[in + j] * state[j];
        }
        sum += sum1 + sum2 + sum3 + sum4;

		r[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
	}
	for (i = 0; i<N; i++)
	{
		/* Compute output. */
		im = i * M;
		in = i * N;
		float sum = gru->bias[2 * N + i];
		// for (j = 0; j<M; j++)
		// 	sum += gru->output_input_weights[im + j] * input[j];
        j = 0;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sum4 = 0.0f;
        while (j + 4 < M) {
            sum1 += gru->output_input_weights[im + j + 0] * input[j + 0];
            sum2 += gru->output_input_weights[im + j + 1] * input[j + 1];
            sum3 += gru->output_input_weights[im + j + 2] * input[j + 2];
            sum4 += gru->output_input_weights[im + j + 3] * input[j + 3];
            j += 4;
        }        
        for (; j < M; j++) {
            sum += gru->output_input_weights[im + j] * input[j];
        }

		// for (j = 0; j<N; j++)
		// 	sum += gru->output_recurrent_weights[in + j] * state[j] * r[j];
        j = 0;
        while (j + 4 < N) {
            sum1 += gru->output_recurrent_weights[in + j + 0] * state[j + 0] * r[j + 0];
            sum2 += gru->output_recurrent_weights[in + j + 1] * state[j + 1] * r[j + 1];
            sum3 += gru->output_recurrent_weights[in + j + 2] * state[j + 2] * r[j + 2];
            sum4 += gru->output_recurrent_weights[in + j + 3] * state[j + 3] * r[j + 3];
            j += 4;
        }        
        for (; j < N; j++) {
            sum += gru->output_recurrent_weights[in + j] * state[j] * r[j];
        }
        sum += sum1 + sum2 + sum3 + sum4;

		if (gru->activation == ACTIVATION_SIGMOID) sum = sigmoid_approx(WEIGHTS_SCALE*sum);
		else if (gru->activation == ACTIVATION_TANH) sum = tansig_approx(WEIGHTS_SCALE*sum);
		else if (gru->activation == ACTIVATION_RELU) sum = relu(WEIGHTS_SCALE*sum);
		//else *(int*)0 = 0;
		h[i] = z[i] * state[i] + (1 - z[i])*sum;
	}
	for (i = 0; i<N; i++)
		state[i] = h[i];
}

void rnnoise_init_rnn(void)
{
	rnnoise_compute_gru = compute_gru;
	rnnoise_compute_dense = compute_dense;
}

