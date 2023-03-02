/*
 *  Copyright (c) 2019 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "pns_fft.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>

namespace pns
{
PnsFft::PnsFft(size_t fftSize) : fftSize_(fftSize)
{
    assert(fftSize == 512 || fftSize == 1024);
    if (fftSize == 512) {
        expSize_ = 9;
    } else {
        expSize_ = 10;
    }

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
}

PnsFft::~PnsFft() { }

void PnsFft::Fft(float* real, float* imag) {
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

void PnsFft::Ifft(float* real, float* imag) {
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

    float modify = 1.0f / (float)fftSize_;
    // std::transform(std::begin(real), std::end(real), std::begin(real), [modify](float a) { return a * modify; });
    // std::transform(std::begin(imag), std::end(imag), std::begin(imag), [modify](float a) { return a * modify; });
    for (size_t i = 0; i < fftSize_; i++) {
        real[i] *= modify;
        imag[i] *= modify;
    }
}

void PnsFft::Shuffle(float* buf)
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

} // namespace pns
