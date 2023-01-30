/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_AUDIO_PROCESSING_HS_HOWLING_SUPPRESSOR_LPC_H_
#define MODULES_AUDIO_PROCESSING_HS_HOWLING_SUPPRESSOR_LPC_H_

#include <memory>
#include <vector>

#define HS_DEBUG_RECORD
#ifdef HS_DEBUG_RECORD
#include <cstdio>
#endif // HS_DEBUG_RECORD

#include "hs_fft_1024.h"
#include "hs_common.h"


namespace webrtc {

class HowlingSuppressorLpc {
public:
    explicit HowlingSuppressorLpc();
    ~HowlingSuppressorLpc();

    // Applies howling suppression.
    void Process(float* audio);

    // Enable or disable howling suppression
    void Enable(bool enable);

    bool isEnabled() { return enabled_; }

private:
    void GetLpc();

    int Predict();

private:
    bool enabled_;

    float howlingSuppression_ = 1.0f;

    std::unique_ptr<HsFft1024> fft_;

    std::vector<float> blockData_;

    std::vector<float> innerProduct_;

    std::vector<float> imagData_;

    std::vector<float> realData_;

    std::vector<float> ampSpecturm_;

    std::vector<float> perceptronModel_;

    std::vector<float> lpcCoeff_;

    std::vector<float> autocorrelationCoeff_;

    std::vector<int> howlingHistory_;

#ifdef HS_DEBUG_RECORD
    FILE* before_hs_ = nullptr;
    FILE* after_hs_ = nullptr;
    FILE* suppression_condition_ = nullptr;
    
    int16_t hs_record_buffer_[kFrameSize];
#endif // HS_DEBUG_RECORD
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_HS_HOWLING_SUPPRESSOR_LPC_H_
