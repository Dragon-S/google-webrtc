#ifndef _NN_PACKET_LOSS_CANCAL_FFT_H_
#define _NN_PACKET_LOSS_CANCAL_FFT_H_


#include <vector>
#include <cstdio>
#include <cmath>

namespace nnPlc
{
class Fft
{
public:
    Fft() = default;
    ~Fft() {}

    int initialize();

    void FftForward(std::vector<float>& real, std::vector<float>& imag);

    void FftBackward(std::vector<float>& real, std::vector<float>& imag);

private:
    void Shuffle(std::vector<float>& data);

private:
    const float pi_ = acosf(-1.0f);

private:
    size_t fftSize_ = 256;

    size_t expSize_ = 8;

    std::vector<size_t> bitReverseList_;

    std::vector<float> realButterflyOperator_;

    std::vector<float> imagBufferflyOperator_;
};
} // namespace nnPlc

#endif // _NN_PACKET_LOSS_CANCAL_FFT_H_