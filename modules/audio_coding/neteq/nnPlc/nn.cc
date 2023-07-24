#include <iostream>
#include <cstdint>
#include <cmath>

#include "weights.h"
#include "nn.h"

namespace nnPlc {

NN::~NN() {
    delete [] encode_;
    delete [] state0_;
    delete [] state1_;
    delete [] rnn_;
    delete [] r_;
    delete [] z_;
}

int NN::initialize() {
    encode_ = new float[EncodeOutputSize_ + 1]; 
    state0_ = new float[RnnOutputSize_];
    state1_ = new float[RnnOutputSize_];
    rnn_ = new float[RnnOutputSize_ + 1];
    r_ = new float[RnnOutputSize_];
    z_ = new float[RnnOutputSize_];

    for (size_t i = 0; i < RnnOutputSize_; i++) {
        state0_[i] = 0.0f;
        state1_[i] = 0.0f;
    }

    return 0;
}

float tansig(float x)
{
    if (isnan(x)) {
        return 0.0f;
    } else if (x < -7.90f) {
        return -1.0f;
    } else if (x > 7.90f) {
        return 1.0f;
    } else {
        float sign = 1.0f;
        if (x < 0.0f) {
            x = -x;
            sign = -1.0f;
        }

        if (x < 3.0f) {
            x = x * 200.0f;
            int i1 = (int)x;
            int i2 = i1 + 1;
            float y = tansigTable1[i1] * ((float)i2 - x) + tansigTable1[i2] * (x - (float)i1);            
            return sign * y;
        } else {
            x = (x - 3.0f) * 20.0f;
            int i1 = (int)x;
            int i2 = i1 + 1;
            float y = tansigTable2[i1] * ((float)i2 - x) + tansigTable2[i2] * (x - (float)i1);            
            return sign * y;            
        }
    }
}

float sigmoid(float x)
{
  return 0.5f + 0.5f * tansig(0.5f * x);
}

void NN::encode(const float* input) {
    for (size_t i = 0; i < EncodeOutputSize_; i++) {
        float sum = bias0[i];
        size_t base = EncodeOutputSize_ * i;
        for (size_t j = 0; j < EncodeInputSize_; j++) {
            sum += weight0[base + j] * input[j];
        }
        encode_[i] = sum < 0.0f ? 0.0f : sum;
    }
}

void NN::tdCapture1() {
	for (size_t i = 0; i < RnnOutputSize_; i++) {
		size_t im = i * RnnInputSize_;
		size_t in = i * RnnOutputSize_;
		float sum = biasIz0[i] + biasHz0[i];
		for (size_t j = 0; j < RnnInputSize_; j++) {
            sum += weightIz0[im + j] * encode_[j];
        }	
		for (size_t j = 0; j < RnnOutputSize_; j++) {
			sum += weightHz0[in + j] * state0_[j];
        }
		z_[i] = sigmoid(sum);
	}
	for (size_t i = 0; i < RnnOutputSize_; i++) {
		size_t im = i * RnnInputSize_;
		size_t in = i * RnnOutputSize_;
		float sum = biasIr0[i] + biasHr0[i];
		for (size_t j = 0; j < RnnInputSize_; j++) {
			sum += weightIr0[im + j] * encode_[j];
        }
		for (size_t j = 0; j < RnnOutputSize_; j++) {
			sum += weightHr0[in + j] * state0_[j];
        }
		r_[i] = sigmoid(sum);
	}
	for (size_t i = 0; i < RnnOutputSize_; i++) {
		size_t im = i * RnnInputSize_;
		size_t in = i * RnnOutputSize_;
		float sum = biasIn0[i];
		for (size_t j = 0; j < RnnInputSize_; j++) {
			sum += weightIn0[im + j] * encode_[j];
        }
		for (size_t j = 0; j < RnnOutputSize_; j++) {
			sum += (weightHn0[in + j] * state0_[j] + biasHn0[i]) * r_[j];
        }
		sum = tansig(sum);
		state0_[i] = z_[i] * state0_[i] + (1 - z_[i]) * sum;
	}
}

void NN::tdCapture2() {
	for (size_t i = 0; i < RnnOutputSize_; i++) {
		size_t in = i * RnnOutputSize_;
		float sum = biasIz1[i] + biasHz1[i];
		for (size_t j = 0; j < RnnOutputSize_; j++) {
            sum += weightIz1[in + j] * state0_[j];
            sum += weightHz1[in + j] * state1_[j];
        }		
		z_[i] = sigmoid(sum);
	}
	for (size_t i = 0; i < RnnOutputSize_; i++) {
		size_t in = i * RnnOutputSize_;
		float sum = biasIr1[i] + biasHr1[i];
		for (size_t j = 0; j < RnnOutputSize_; j++) {
			sum += weightIr1[in + j] * state0_[j];
			sum += weightHr1[in + j] * state1_[j];
        }
		r_[i] = sigmoid(sum);
	}
	for (size_t i = 0; i < RnnOutputSize_; i++) {
		size_t in = i * RnnOutputSize_;
		float sum = biasIn1[i];
		for (size_t j = 0; j < RnnOutputSize_; j++) {
			sum += weightIn1[in + j] * state0_[j];
			sum += (weightHn1[in + j] * state1_[j] + biasHn1[i]) * r_[j];
        }
		sum = tansig(sum);
		state1_[i] = z_[i] * state1_[i] + (1 - z_[i]) * sum;
	}
}

void NN::decode(float* output) {
    for (size_t i = 0; i < DecodeOutputSize_; i++) {
        float sum = bias1[i];
        size_t base = DecodeOutputSize_ * i;
        for (size_t j = 0; j < DecodeInputSize_; j++) {
            sum += weight1[base + j] * rnn_[j];
        }
        output[i] = sigmoid(sum);
    }
}

void NN::process(const float* input, float* output) {
    float ticker = input[EncodeInputSize_ - 1];

    // encode
    encode(input);
    encode_[EncodeOutputSize_] = ticker;

    // RNN
    tdCapture1();
    tdCapture2();
    for (size_t i = 0; i < RnnOutputSize_; i++) {
        rnn_[i] = state1_[i];
    }
    rnn_[RnnOutputSize_] = ticker;

    // decode
    decode(output);
}

} // namespace nnPlc