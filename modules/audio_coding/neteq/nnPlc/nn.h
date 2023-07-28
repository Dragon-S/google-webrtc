#ifndef _NN_PACKET_LOSS_CANCAL_NN_H_
#define _NN_PACKET_LOSS_CANCAL_NN_H_

#include <iostream>
#include <cstdint>

namespace nnPlc
{ 
class NN {
public:
    NN() = default;
    ~NN();

    int initialize();

    void process(const float* input, float* output);

private:
    void encode(const float* input);
    void tdCapture1();
    void tdCapture2();
    void decode(float* output);

    float tansig(float x);
    float sigmoid(float x);

private:
    const size_t EncodeInputSize_ = 129;
    const size_t EncodeOutputSize_ = 64;
    const size_t RnnInputSize_ = 65;
    const size_t RnnOutputSize_ = 64;
    const size_t DecodeInputSize_ = 65;
    const size_t DecodeOutputSize_ = 128;

    float *encode_ = nullptr;
    float *state0_ = nullptr;
    float *state1_ = nullptr;
    float *rnn_ = nullptr;

    float *z_ = nullptr;
    float *r_ = nullptr;
    float *s_ = nullptr;
};

} // namespace nnPlc

#endif // _NN_PACKET_LOSS_CANCAL_NN_H_