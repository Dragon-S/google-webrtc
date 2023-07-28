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
    delete [] s_;
}

int NN::initialize() {
    encode_ = new float[EncodeOutputSize_ + 1]; 
    state0_ = new float[RnnOutputSize_];
    state1_ = new float[RnnOutputSize_];
    rnn_ = new float[RnnOutputSize_ + 1];
    r_ = new float[RnnOutputSize_];
    z_ = new float[RnnOutputSize_];
    s_ = new float[RnnOutputSize_];

    for (size_t i = 0; i < RnnOutputSize_; i++) {
        state0_[i] = 0.0f;
        state1_[i] = 0.0f;
    }

    return 0;
}

float NN::tansig(float x) {
   int i;
   float y, dy;
   float sign = 1.0f;

   //std::cout << x << ", ";

   if (isnan(x)) {
       //std::cout << "0.0" << std::endl;
       return 0.0f;
   }

   if (!(x < 8.0f)) {
       //std::cout << "1.0" << std::endl;
       return 1.0f;
   }else if (!(x > -8.0f)) {
       //std::cout << "-1.0" << std::endl;
       return -1.0f;
   } else {
       if (x < 0.0f) {
           x = -x;
           sign = -1.0f;
       }
       i = (int)floor(0.5f + 25.0f * x);
       x -= 0.04f * i;
       y = tansigTable[i];
       dy = 1.0f - y * y;
       y = y + x * dy * (1.0f - y * x);
       //std::cout << sign * y << std::endl;
       return sign * y;
   }
}

float NN::sigmoid(float x) {
  return 0.5f + 0.5f * tansig(0.5f * x);
}

void NN::encode(const float* input) {
    for (size_t i = 0; i < EncodeOutputSize_; i++) {
        float sum = bias0[i];
        size_t base = EncodeInputSize_ * i;
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
		float sumI = biasIn0[i];
        float sumH = biasHn0[i];
		for (size_t j = 0; j < RnnInputSize_; j++) {
			sumI += weightIn0[im + j] * encode_[j];
        }
		for (size_t j = 0; j < RnnOutputSize_; j++) {
            sumH += weightHn0[in + j] * state0_[j];
        }
        float sum = tansig(sumI + sumH * r_[i]);
		s_[i] = z_[i] * state0_[i] + (1 - z_[i]) * sum;
	}

    //for (size_t i = 0; i < RnnOutputSize_; i++) {
    //    std::cout << s_[i] << ", ";
    //    if ((i + 1) % 4 == 0) {
    //        std::cout << std::endl;
    //    }
    //}

    for (size_t i = 0; i < RnnOutputSize_; i++) {
        state0_[i] = s_[i];
    }
}

void NN::tdCapture2() {
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
		float sum = biasIz1[i] + biasHz1[i];
		for (size_t j = 0; j < RnnOutputSize_; j++) {
            sum += weightIz1[in + j] * state0_[j];
            sum += weightHz1[in + j] * state1_[j];
        }		
		z_[i] = sigmoid(sum);
	}
	for (size_t i = 0; i < RnnOutputSize_; i++) {
		size_t in = i * RnnOutputSize_; 
		float sumI = biasIn1[i];
        float sumH = biasHn1[i];
		for (size_t j = 0; j < RnnOutputSize_; j++) {
			sumI += weightIn1[in + j] * state0_[j];
			sumH += weightHn1[in + j] * state1_[j];
        }
		float sum = tansig(sumI + sumH * r_[i]);
		s_[i] = z_[i] * state1_[i] + (1 - z_[i]) * sum;
	}

    //for (size_t i = 0; i < RnnOutputSize_; i++) {
    //    std::cout << s_[i] << ", ";
    //    if ((i + 1) % 4 == 0) {
    //        std::cout << std::endl;
    //    }
    //}

    for (size_t i = 0; i < RnnOutputSize_; i++) {
        state1_[i] = s_[i];
    }
}

void NN::decode(float* output) {
    for (size_t i = 0; i < DecodeOutputSize_; i++) {
        float sum = bias1[i];
        size_t base = DecodeInputSize_ * i;
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
    //for (size_t i = 0; i < DecodeOutputSize_; i++) {
    //    std::cout << output[i] << ", ";
    //    if ((i + 1) % 6 == 0) {
    //        std::cout << std::endl;
    //    }
    //}
    //std::cout << std::endl;
}

} // namespace nnPlc