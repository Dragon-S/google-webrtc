/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "howling_suppressor_lpc.h"
#include "hs_fft_1024.h"
#include "hs_common.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

#include "rtc_base/logging.h"

namespace webrtc {

HowlingSuppressorLpc::HowlingSuppressorLpc() :
    enabled_(false),
    fft_(new HsFft1024()) { 

    blockData_.resize(kHsFftSize);
    innerProduct_.resize(kHsFftSize - kHsLpcOrder);
    imagData_.resize(kHsFftSize);
    realData_.resize(kHsFftSize);
    ampSpecturm_.resize(kHsFftSizeBy2);
    lpcCoeff_.resize(kHsLpcOrder + 1);
    autocorrelationCoeff_.resize(kHsLpcOrder + 1);
    howlingHistory_.clear();
    perceptronModel_.assign(std::begin(kHsPerceptronModel), std::end(kHsPerceptronModel));

#ifdef HS_DEBUG_RECORD
    before_hs_ = fopen("/sdcard/webrtc3Arecord/before_hs.pcm", "wb");
    after_hs_ = fopen("/sdcard/webrtc3Arecord/after_hs.pcm", "wb");
    suppression_condition_ = fopen("/sdcard/webrtc3Arecord/suppression_condition.txt", "wb");
#endif // HS_DEBUG_RECORD
}

HowlingSuppressorLpc::~HowlingSuppressorLpc() {
    fft_.reset(nullptr);

    blockData_.clear();
    imagData_.clear();
    realData_.clear();
    ampSpecturm_.clear();
    lpcCoeff_.clear();
    autocorrelationCoeff_.clear();
    howlingHistory_.clear();
    perceptronModel_.clear();
#ifdef HS_DEBUG_RECORD
    fclose(before_hs_);
    fclose(after_hs_);
    fclose(suppression_condition_);
#endif // HS_DEBUG_RECORD
}

void HowlingSuppressorLpc::GetLpc() {
    // auto correlation
    for (size_t i = 0; i < kHsLpcOrder + 1; i++) {
        float x = 0.0f;
        std::transform(blockData_.begin() + kHsLpcOrder + 1, blockData_.end(), blockData_.begin() + kHsLpcOrder - i + 1, innerProduct_.begin(), [](float a, float b) { return a * b; });
        std::for_each(innerProduct_.begin(), innerProduct_.end(), [&x](float a) { x += a; });
        autocorrelationCoeff_[i] = x;
    }

    // lpc estimate
    bool fineEstimate = true;
    float error = autocorrelationCoeff_[0];
    float lpcMatrix[kHsLpcOrder + 1][kHsLpcOrder + 1] = {0.0f};
    for (size_t i = 1; i < kHsLpcOrder + 1; i++) {
        if (fabsf(error) < 0.0001f) {
            fineEstimate = false;
            break;
        }

        float temp = autocorrelationCoeff_[i];
        for (size_t j = 1; j < i; j++) {
            temp -= (lpcMatrix[i - 1][j] * autocorrelationCoeff_[i - j]);
        }
        temp /= error;
        lpcMatrix[i][i] = temp;

        if (i > 1) {
            for (size_t j = 1; j < i; j++) {
                lpcMatrix[i][j] = lpcMatrix[i - 1][j] - temp * lpcMatrix[i - 1][i - j]; 
            }
        }

        error *= (1.0f - temp * temp);
    }

    lpcCoeff_[0] = 1.0f;
    if (fineEstimate) {
        for (size_t i = 1; i < kHsLpcOrder + 1; i++) {
            lpcCoeff_[i] = -lpcMatrix[kHsLpcOrder][i];
        }
    } else {
        for (size_t i = 1; i < kHsLpcOrder + 1; i++) {
            lpcCoeff_[i] = 0.0f;
        }
    }
}

int HowlingSuppressorLpc::Predict() {
    float perceptron = 9.0f;
    std::transform(ampSpecturm_.begin(), ampSpecturm_.end(), perceptronModel_.begin(), ampSpecturm_.begin(), [](float a, float b) { return a * b; });
    std::for_each(ampSpecturm_.begin(), ampSpecturm_.end(), [&perceptron](float a) { perceptron += a; });
    return perceptron > 0.0f ? 1 : 0;
}

void HowlingSuppressorLpc::Process(float* audio) {
    if (!enabled_) {
        return;
    }

#ifdef HS_DEBUG_RECORD
    if (before_hs_) {
        for (size_t i = 0; i < kFrameSize; i++) {
            hs_record_buffer_[i] = (int16_t)audio[i];
        }
        fwrite(&hs_record_buffer_[0], sizeof(int16_t), kFrameSize, before_hs_);
    }
#endif // HS_DEBUG_RECORD

    // get origin data
    for (size_t i = 0; i < kHsFftSize - kFrameSize; i++) {
        blockData_[i] = blockData_[kFrameSize + i];
    }
    for (size_t i = 0; i < kFrameSize; i++) {
        blockData_[kHsFftSize - kFrameSize + i] = audio[i];
    }

    // calculate frame energy
    float energy = 0.0f;
    std::for_each(blockData_.begin(), blockData_.end(), [&energy](float a) { energy += (a * a); });

    int label = 0;
    if (energy >= 5.76e6) {
        // calculate lpc
        GetLpc();

        // burge specturm estimate
        for (size_t i = 0; i < kHsFftSize; i++) {
            realData_[i] = 0.0f;
            imagData_[i] = 0.0f;
        }
        std::copy(lpcCoeff_.begin(), lpcCoeff_.end(), realData_.begin());
        fft_->Fft(realData_, imagData_);
        std::transform(realData_.begin(), realData_.begin() + kHsFftSizeBy2, imagData_.begin(), ampSpecturm_.begin(), [](float a, float b) { return 20.0f * log10f(1.0f / (sqrtf(a * a + b * b) + 0.0001f) + 1.0f); });

        // specturm normalize
        float min = ampSpecturm_[0];
        float max = ampSpecturm_[0]; 
        for (size_t i = 1; i < kHsFftSizeBy2; i++) {
            min = min > ampSpecturm_[i] ? ampSpecturm_[i] : min;
            max = max < ampSpecturm_[i] ? ampSpecturm_[i] : max;
        }
        std::transform(ampSpecturm_.begin(), ampSpecturm_.end(), ampSpecturm_.begin(), [min, max](float a) { return (a - min) / (max - min + 0.0001f); });

        // preceptron predict result
        label = Predict();   

        // time domain estimate
        howlingHistory_.push_back(label);
        if (howlingHistory_.size() > 100) {
            howlingHistory_.erase(howlingHistory_.begin());
        }
        int howlingCount = 0;
        std::for_each(howlingHistory_.begin(), howlingHistory_.end(), [&howlingCount](int a) { howlingCount += a; });
        if (howlingHistory_.size() == 100 && howlingCount <= 50) {
            label = 0;
        }
    }

    float targetSuppression = howlingSuppression_;
    if (energy >= 5.76e6) {
        if (label) {
            targetSuppression = howlingSuppression_ * kAttackFactor;
        } else {
            targetSuppression = howlingSuppression_ * kReleaseFactor + (1.0f - kReleaseFactor);
        }
    }

    float step = (targetSuppression - howlingSuppression_) / (float)kFrameSize;
    for (size_t i = 0; i < kFrameSize; i++) {
        audio[i] *= howlingSuppression_;
        howlingSuppression_ += step;
    }

#ifdef HS_DEBUG_RECORD
    if (after_hs_) {
        for (size_t i = 0; i < kFrameSize; i++) {
            hs_record_buffer_[i] = (int16_t)audio[i];
        }
        fwrite(&hs_record_buffer_[0], sizeof(int16_t), kFrameSize, after_hs_);
    }
#endif // HS_DEBUG_RECORD
}

void HowlingSuppressorLpc::Enable(bool enable) {
    enabled_ = enable;
}

}  // namespace webrtc
