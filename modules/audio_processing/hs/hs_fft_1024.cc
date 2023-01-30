/*
 *  Copyright (c) 2019 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "hs_fft_1024.h"
#include "hs_common.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace webrtc
{
HsFft1024::HsFft1024()
{
    auto bitReverse = [](size_t n) {
        size_t i = 0;
        size_t r = 0;
        do
        {
            r |= (n & 0x0001);
            n >>= 1;
            r <<= 1;
        } while (i++ < (kHsFftExp - 2));
        r |= (n & 0x0001);
        return r;
    };

    bitReverseList_.resize(kHsFftSize);
    realButterflyOperator_.resize(kHsFftSize);
    imagBufferflyOperator_.resize(kHsFftSize);
    for (size_t i = 0; i < kHsFftSize; i++)
    {
        bitReverseList_[i] = bitReverse(i);
        realButterflyOperator_[i] = cosf(2.0f * kMPi * i / (float)kHsFftSize);
        imagBufferflyOperator_[i] = sinf(2.0f * kMPi * i / (float)kHsFftSize);
    }
}

HsFft1024::~HsFft1024() { }

void HsFft1024::Fft(std::vector<float> &real, std::vector<float> &imag) {
    Shuffle(real);

    for (size_t i = 0; i < kHsFftExp; i++)
    {
        size_t bs = 1 << (kHsFftExp - 1 - i);
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

void HsFft1024::Shuffle(std::vector<float> &buf)
{
    std::vector<bool> flag(kHsFftSize, true);
    for (size_t i = 0; i < kHsFftSize; i++)
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

} // namespace webrtc
