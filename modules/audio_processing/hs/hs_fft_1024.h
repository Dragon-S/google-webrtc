/*
 *  Copyright (c) 2019 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef MODULES_AUDIO_PROCESSING_HS_HS_FFT_H_
#define MODULES_AUDIO_PROCESSING_HS_HS_FFT_H_

#include "hs_common.h"

#include <vector>
#include <cstdio>

namespace webrtc
{
class HsFft1024
{
public:
    explicit HsFft1024();
    ~HsFft1024();

    void Fft(std::vector<float> &real, std::vector<float> &imag);

private:
    void Shuffle(std::vector<float> &data);

private:
    std::vector<size_t> bitReverseList_;

    std::vector<float> realButterflyOperator_;

    std::vector<float> imagBufferflyOperator_;
};
} // namespace webrtc

#endif // MODULES_AUDIO_PROCESSING_HS_HS_FFT_H_