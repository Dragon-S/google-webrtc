/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_AUDIO_PROCESSING_HS_HOWLING_SUPPRESSOR_H_
#define MODULES_AUDIO_PROCESSING_HS_HOWLING_SUPPRESSOR_H_

#include <memory>
#include <vector>
#include <utility>

// #define HS_DEBUG_RECORD
#ifdef HS_DEBUG_RECORD
#include <cstdio>
#endif // HS_DEBUG_RECORD

#include "hs_fft_1024.h"
#include "hs_common.h"
#include "notch_filter.h"


namespace webrtc {

class HowlingSuppressor {
public:
    explicit HowlingSuppressor();
    ~HowlingSuppressor();

    // Applies howling suppression.
    void Process(float* audio);

    // Enable or disable howling suppression
    void Enable(bool enable);

    bool isEnasbled() { return enabled_; }

private:
    bool enabled_;

    float howlingCount_ = 0.0f;

    float howlingSuppression_ = 1.0f;

    float quantilyRate_ = 0.6f;

    float highQuantilyRate_ = 0.6f;

    std::unique_ptr<HsFft1024> fft_;

    std::vector<float> fftData_;

    std::vector<float> imagData_;

    std::vector<float> realData_;

    std::vector<float> powSpectrum_;

    std::vector<int> voiceHistory_;

    std::vector<size_t> maybeHowlingHistory_;

    std::vector<size_t> howlingContinue_;

    std::vector<std::pair<size_t, std::unique_ptr<NotchFilter> > > notchFilter_;

#ifdef HS_DEBUG_RECORD
    FILE* before_hs_ = nullptr;
    FILE* after_hs_ = nullptr;
    FILE* suppression_condition_ = nullptr;
    
    int16_t hs_record_buffer_[kHsFrameSize];
#endif // HS_DEBUG_RECORD
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_HS_HOWLING_SUPPRESSOR_H_
