/*
 *  Copyright (c) 2019 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef PNS_FFT_H_
#define PNS_FFT_H_


#include <vector>
#include <cstdio>
#include <cmath>

namespace pns
{
class PnsFft
{
public:
    explicit PnsFft(size_t fftSize);
    ~PnsFft();

    void Fft(float* real, float* imag);

    void Ifft(float* real, float* imag);

private:
    void Shuffle(float* data);

private:
    const float pi_ = acosf(-1.0f);

private:
    size_t fftSize_ = 256;

    size_t expSize_ = 8;

    std::vector<size_t> bitReverseList_;

    std::vector<float> realButterflyOperator_;

    std::vector<float> imagBufferflyOperator_;
};
} // namespace pns

#endif // PNS_FFT_H_