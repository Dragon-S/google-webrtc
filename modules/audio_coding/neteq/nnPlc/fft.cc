#include "fft.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>

namespace nnPlc
{
int Fft::initialize()
{
    auto bitReverse = [](size_t n, size_t expSize) {
        size_t i = 0;
        size_t r = 0;
        do
        {
            r |= (n & 0x0001);
            n >>= 1;
            r <<= 1;
        } while (i++ < (expSize - 2));
        r |= (n & 0x0001);
        return r;
    };

    bitReverseList_.resize(fftSize_);
    realButterflyOperator_.resize(fftSize_);
    imagBufferflyOperator_.resize(fftSize_);
    for (size_t i = 0; i < fftSize_; i++)
    {
        bitReverseList_[i] = bitReverse(i, expSize_);
        realButterflyOperator_[i] = cosf(2.0f * pi_ * i / (float)fftSize_);
        imagBufferflyOperator_[i] = sinf(2.0f * pi_ * i / (float)fftSize_);
    }

    return 0;
}

void Fft::FftForward(std::vector<float>& real, std::vector<float>& imag) {
    Shuffle(real);

    for (size_t i = 0; i < expSize_; i++)
    {
        size_t bs = 1 << (expSize_ - 1 - i);
        size_t ps = 1 << i;
        for (size_t j = 0; j < bs; j++)
        {
            for (size_t k = 0; k < ps; k++)
            {
                size_t position = bs * k;
                float realRotation = realButterflyOperator_[position];
                float imagRotation = -imagBufferflyOperator_[position];

                size_t position1 = 2 * j * ps + k;
                size_t position2 = 2 * j * ps + ps + k;

                float realPart = real[position2] * realRotation - imag[position2] * imagRotation;
                float imagPart = real[position2] * imagRotation + imag[position2] * realRotation;
                real[position2] = real[position1] - realPart;
                imag[position2] = imag[position1] - imagPart;
                real[position1] = real[position1] + realPart;
                imag[position1] = imag[position1] + imagPart;
            }
        }
    }
}

void Fft::FftBackward(std::vector<float>& real, std::vector<float>& imag) {
    Shuffle(real);
    Shuffle(imag);

    for (size_t i = 0; i < expSize_; i++)
    {
        size_t bs = 1 << (expSize_ - 1 - i);
        size_t ps = 1 << i;
        for (size_t j = 0; j < bs; j++)
        {
            for (size_t k = 0; k < ps; k++)
            {
                size_t position = bs * k;
                float realRotation = realButterflyOperator_[position];
                float imagRotation = imagBufferflyOperator_[position];

                size_t position1 = 2 * j * ps + k;
                size_t position2 = 2 * j * ps + ps + k;

                float realPart = real[position2] * realRotation - imag[position2] * imagRotation;
                float imagPart = real[position2] * imagRotation + imag[position2] * realRotation;
                real[position2] = real[position1] - realPart;
                imag[position2] = imag[position1] - imagPart;
                real[position1] = real[position1] + realPart;
                imag[position1] = imag[position1] + imagPart;
            }
        }
    }

    std::transform(real.begin(), real.end(), real.begin(), [](float a) { return a / 256.0f; });
    std::transform(imag.begin(), imag.end(), imag.begin(), [](float a) { return a / 256.0f; });
}

void Fft::Shuffle(std::vector<float>& buf)
{
    std::vector<bool> flag(fftSize_, true);
    for (size_t i = 0; i < fftSize_; i++)
    {
        if (flag[i])
        {
            float temp = buf[i];
            buf[i] = buf[bitReverseList_[i]];
            buf[bitReverseList_[i]] = temp;
            flag[i] = false;
            flag[bitReverseList_[i]] = false;
        }
    }
}

} // namespace nnPlc
