/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "howling_suppressor.h"
#include "hs_fft_1024.h"
#include "hs_common.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

namespace webrtc {

HowlingSuppressor::HowlingSuppressor() :
    enabled_(false),
    fft_(new HsFft1024()) { 

    fftData_.resize(kHsFftSize);
    imagData_.resize(kHsFftSize);
    realData_.resize(kHsFftSize);
    powSpectrum_.resize(kHsFftSizeBy2Plus1);
    voiceHistory_.clear();
    maybeHowlingHistory_.resize(4, 0);
    howlingContinue_.resize(kHsFftSizeBy2 - kHsFftSizeBy2 / 8, 0);
    notchFilter_.resize(kMaxNotchFilter);
    for (size_t i = 0; i < kMaxNotchFilter; i++) {
        notchFilter_[i].first = 0;
        notchFilter_[i].second.reset(new NotchFilter(0, 1.0f));
    }

#ifdef HS_DEBUG_RECORD
#ifdef LOCAL_TEST
    suppression_condition_ = fopen("samples/suppression_condition.pcm", "wb");
#else
    before_hs_ = fopen("./before_hs.pcm", "wb");
    after_hs_ = fopen("./after_hs.pcm", "wb");
    suppression_condition_ = fopen("./suppression_condition.pcm", "wb");    
#endif // LOCAL_TEST
#endif // HS_DEBUG_RECORD
}

HowlingSuppressor::~HowlingSuppressor() {
    fft_.reset(nullptr);

    fftData_.clear();
    imagData_.clear();
    realData_.clear();
    powSpectrum_.clear();
    voiceHistory_.clear();
    maybeHowlingHistory_.clear();
    howlingContinue_.clear();
    for (size_t i = 0; i < kMaxNotchFilter; i++) {
        notchFilter_[i].first = 0;
        notchFilter_[i].second.release();
    }    
    notchFilter_.clear();

#ifdef HS_DEBUG_RECORD
    fclose(before_hs_);
    fclose(after_hs_);
    fclose(suppression_condition_);
#endif // HS_DEBUG_RECORD
}

void HowlingSuppressor::Process(float* audio, float* audioHighband) {
    if (!enabled_) {
        return;
    }

#ifdef HS_DEBUG_RECORD
    if (before_hs_) {
        for (size_t i = 0; i < kHsFrameSize; i++) {
            hs_record_buffer_[i] = (int16_t)audio[i];
        }
        fwrite(&hs_record_buffer_[0], sizeof(int16_t), kHsFrameSize, before_hs_);
    }
#endif // HS_DEBUG_RECORD

    //std::copy(audio.begin(), audio.end(), fftData_.begin() + kHsFftSize - kHsFrameSize);
    for (size_t i = 0; i < kHsFftSize - kHsFrameSize; i++) {
        fftData_[i] = fftData_[kHsFrameSize + i];
    }
    for (size_t i = 0; i < kHsFrameSize; i++) {
        fftData_[kHsFftSize - kHsFrameSize + i] = audio[i];
    }
    std::copy(fftData_.begin(), fftData_.end(), realData_.begin());
    for (size_t i = 0; i < kHsFftSize; i++) {
        imagData_[i] = 0.0f;
    }
    fft_->Fft(realData_, imagData_);
    std::transform(realData_.begin(), realData_.begin() + kHsFftSizeBy2Plus1, imagData_.begin(), powSpectrum_.begin(), [](float a, float b) { return sqrtf(a * a + b * b); });
    //float meanAmp = std::accumulate(powSpectrum_.cbegin(), powSpectrum_.cend(), 0.0f) / (float)kHsFftSizeBy2Plus1;
    float totalAmp = 0.0f;
    float meanAmp = 0.0f;
    for (size_t i = 0; i < kHsFftSizeBy2Plus1; i++) {
        totalAmp += powSpectrum_[i];
    }
    meanAmp = totalAmp / (float)kHsFftSizeBy2Plus1;

    int lowPeaks = 0;
    int highPeaks = 0;
    int howlingCount = 0;
    int firstPeak = -1;
    for (size_t i = 1; i < kHsFftSizeBy2Plus1 - 1; i++) {
        // peaks with threshold
        float rate1 = log2f(powSpectrum_[i] / (meanAmp + 0.0001f) + 1.0f);
        if (rate1 > kHowlingJudgeThreshold1) {
            // find out the frequency-domain peak point
            float rate2L = log2f(powSpectrum_[i] / (powSpectrum_[i - 1] + 0.0001f) + 1.0f);
            float rate2R = log2f(powSpectrum_[i] / (powSpectrum_[i + 1] + 0.0001f) + 1.0f);
            if ((rate2L > 1.0f) && (rate2R > 1.0f)) {
                if (i < kHsFftSize / 4) {
                    lowPeaks += 1;
                } else {
                    highPeaks += 1;
                }
                if (firstPeak == -1) {
                    firstPeak = (int)i;
                }
                if (rate1 > kHowlingJudgeThreshold2) {
                    howlingCount_ += 1;
                }
            }  
        }
        if ((i >= kHsFftSizeBy2 / 8) && (totalAmp > 2000.0f)) {
            if (rate1 > kHowlingJudgeThreshold7) {
                howlingContinue_[i - kHsFftSizeBy2 / 8] += 1;
            } else {
                howlingContinue_[i - kHsFftSizeBy2 / 8] /= 2;
            }
        }
    }

    float quantilyEnergy = 0.0f;
    float totalEnergy = 0.0f;
    std::vector<float> sortedPowSpecturm(kHsFftSize / 4, 0.0f);
    std::copy(powSpectrum_.begin(), powSpectrum_.begin() + kHsFftSize / 4, sortedPowSpecturm.begin());
    std::sort(sortedPowSpecturm.begin(), sortedPowSpecturm.end());
    std::transform(sortedPowSpecturm.begin(), sortedPowSpecturm.end(), sortedPowSpecturm.begin(), [](float a) { return a * a; });
    for (size_t i = 0; i < kHsFftSize / 4; i++) {
        totalEnergy += sortedPowSpecturm[i];
        if (i >= kHsFftSize / 4 - 6) {
            quantilyEnergy += sortedPowSpecturm[i];
        }
    }
    quantilyRate_ = 0.9f * quantilyRate_ + 0.1f * quantilyEnergy / (totalEnergy + 0.0001f);
    quantilyRate_ = quantilyRate_ > 1.0f ? 1.0f : quantilyRate_; 

    float highQuantilyEnergy = 0.0f;
    float highTotalEnergy = 0.0f;
    std::vector<float> highSortedPowSpecturm(kHsFftSize / 4, 0.0f);
    std::copy(powSpectrum_.begin() + kHsFftSize / 4, powSpectrum_.begin() + kHsFftSize / 2, highSortedPowSpecturm.begin());
    std::sort(highSortedPowSpecturm.begin(), highSortedPowSpecturm.end());
    std::transform(highSortedPowSpecturm.begin(), highSortedPowSpecturm.end(), highSortedPowSpecturm.begin(), [](float a) { return a * a; });
    for (size_t i = 0; i < kHsFftSize / 4; i++) {
        highTotalEnergy += highSortedPowSpecturm[i];
        if (i >= kHsFftSize / 4 - 6) {
            highQuantilyEnergy += highSortedPowSpecturm[i];
        }
    }
    highQuantilyRate_ = 0.9f * highQuantilyRate_ + 0.1f * highQuantilyEnergy / (highTotalEnergy + 0.0001f);
    highQuantilyRate_ = highQuantilyRate_ > 1.0f ? 1.0f : highQuantilyRate_;

    bool condition1 = howlingCount >= kHowlingJudgeThreshold4 && howlingCount <= kHowlingJudgeThreshold3;
    bool condition2 = (lowPeaks < highPeaks) && (highQuantilyRate_ > kHowlingJudgeThreshold6);
    bool condition3 = ((size_t)firstPeak > kHsFftSize / 32) && (highQuantilyRate_ > kHowlingJudgeThreshold6);
    bool condition4 = ((size_t)firstPeak > kHsFftSize / 32) && (quantilyRate_ > kHowlingJudgeThreshold5);

    if  (highTotalEnergy + totalEnergy > 2000.0f) {
        if (!(condition1 || condition2 || condition3 || condition4)) {
            voiceHistory_.push_back(1);
        } else {
            voiceHistory_.push_back(0);
        }
        if (voiceHistory_.size() >= 100) {
            voiceHistory_.erase(voiceHistory_.begin());
        }
    }

    int voiceCount = 0;
    for_each(voiceHistory_.begin(), voiceHistory_.end(), [&voiceCount](int x) { voiceCount += x; });

    // bool hasHowling = false;
    bool maybeHowling = false;

    std::vector<size_t> howlingIndex;
    howlingIndex.clear();
    // howling already exist, do compression
    if (voiceCount < 40) {
        // hasHowling = true;
        float targetSuppression = howlingSuppression_ * kAttackFactor;
        if (targetSuppression < 0.01f) {
            targetSuppression = 0.01f;
        }
        float step = (targetSuppression - howlingSuppression_) / (float)kHsFrameSize;
        for (size_t i = 0; i < kHsFrameSize; i++) {
            audio[i] *= howlingSuppression_;
            if (audioHighband != nullptr) {
                audioHighband[i] *= howlingSuppression_;
            }
            howlingSuppression_ += step;
        }
    } else {
        float targetSuppression = howlingSuppression_ * kReleaseFactor + (1.0f - kReleaseFactor);
        float step = (targetSuppression - howlingSuppression_) / (float)kHsFrameSize;
        for (size_t i = 0; i < kHsFrameSize; i++) {
            audio[i] *= howlingSuppression_;
            if (audioHighband != nullptr) {
                audioHighband[i] *= howlingSuppression_;
            }
            howlingSuppression_ += step;
        }

        // maybe appear howling, do filter
        size_t startHowling = 0;
        size_t endHowling = 0;
        for (size_t i = 1; i < kHsFftSizeBy2 - kHsFftSizeBy2 / 8; i++) {
            if (howlingContinue_[i] > 0) {
                if (startHowling == 0) {
                    startHowling = i;
                } else {
                    endHowling = i;
                }
            } else {
                if (endHowling > 0) {
                    float localAmp = 0.0f;
                    for (size_t j = startHowling; j <= endHowling; j++) {
                        localAmp += powSpectrum_[j + kHsFftSizeBy2 / 8];
                    }
                    size_t howlingWidth = endHowling - startHowling + 1;
                    localAmp /= (float)howlingWidth;

                    float lowerLocalAmp = 0.0f;
                    size_t lowerStart = 0;
                    size_t lowerWidth = startHowling + kHsFftSizeBy2 / 8;
                    if ((startHowling + kHsFftSizeBy2 / 8) > (howlingWidth * 2)) {
                        lowerStart = startHowling + kHsFftSizeBy2 / 8 - howlingWidth * 2;
                        lowerWidth = howlingWidth * 2;
                    }
                    for (size_t j = lowerStart; j < startHowling; j++) {
                        lowerLocalAmp += powSpectrum_[j];
                    }
                    lowerLocalAmp /= (float)lowerWidth;

                    float upperLocalAmp = 0.0f;
                    size_t upperEnd = kHsFftSizeBy2;
                    size_t upperWidth = 7 * kHsFftSizeBy2 / 8 - endHowling;
                    if ((7 * kHsFftSizeBy2 / 8 - endHowling) > (howlingWidth * 2)) {
                        upperEnd = endHowling + kHsFftSizeBy2 / 8 + howlingWidth * 2;
                        upperWidth = howlingWidth * 2;
                    }
                    for (size_t j = endHowling + 1; j <= upperEnd; j++) {
                        upperLocalAmp += powSpectrum_[j];
                    }
                    upperLocalAmp /= (float)upperWidth;

                    float lowerRate = log2f(localAmp / (lowerLocalAmp + 0.0001f) + 1.0f);
                    float upperRate = log2f(localAmp / (upperLocalAmp + 0.0001f) + 1.0f);
                    float localRate = log2f(localAmp / (meanAmp + 0.0001f) + 1.0f);
                    if ((howlingWidth >= 4) && (localRate >= kHowlingJudgeThreshold2) && (lowerRate >= 1.1375f) && (upperRate >= 1.1375f)) {
                        howlingIndex.push_back(kHsFftSizeBy2 / 8 + (endHowling + startHowling) / 2);
                    }
                }
                startHowling = 0;
                endHowling = 0;
            } 
        }
        if ((howlingIndex.size() < 4) && (howlingIndex.size() > 0)) {
            maybeHowling = true;
            maybeHowlingHistory_.push_back(1);
        } else {
            howlingIndex.clear();
            maybeHowlingHistory_.push_back(0);
        }
        if (maybeHowlingHistory_.size() > 4) {
            maybeHowlingHistory_.erase(maybeHowlingHistory_.begin());
        }
        size_t maybeHowlingCount = maybeHowlingHistory_[0] + maybeHowlingHistory_[1] + maybeHowlingHistory_[2] + maybeHowlingHistory_[3];

        if ((maybeHowlingCount < 2) || maybeHowling) {
            for (size_t i = 0; i < kMaxNotchFilter; i++) {
                bool indexInFilter = false;
                size_t currentIndex = notchFilter_[i].first;
                for (size_t j = 0; j < howlingIndex.size(); j++) {
                    if (currentIndex == howlingIndex[j]) {
                        indexInFilter = true;
                        howlingIndex.erase(howlingIndex.begin() + j);
                    }
                }
                if ((!indexInFilter) && (currentIndex > 0)) {
                    notchFilter_[i].first += kFadeOutMark;
                }
            }
        }

        for (size_t i = 0; i < howlingIndex.size(); i++) {
            for (size_t j = 0; j < kMaxNotchFilter; j++) {
                if (notchFilter_[j].first == 0) {
                    notchFilter_[j].first = howlingIndex[i] + kFadeInMark;
                    notchFilter_[j].second->redesign(howlingIndex[i], 0.0001f, 1.0f);
                    break;
                }
            }
        }

        float beforeEnergy = 0.0f;
        // std::for_each(std::begin(audio), std::end(audio), [&beforeEnergy](float x) { beforeEnergy += x * x; });
        for (size_t i = 0; i < kHsFrameSize; i++) {
            beforeEnergy += (audio[i] * audio[i]);
        }

        for (size_t i = 0; i < kMaxNotchFilter; i++) {
            if (notchFilter_[i].first > kFadeOutMark) {
                notchFilter_[i].first = 0;
                notchFilter_[i].second->runningFilter(audio, kHsFrameSize, filterFade::kFadeOut);
            } else if (notchFilter_[i].first > kFadeInMark) {
                notchFilter_[i].first -= kFadeInMark;
                notchFilter_[i].second->runningFilter(audio, kHsFrameSize, filterFade::kFadeIn);
            } else if (notchFilter_[i].first > 0) {
                notchFilter_[i].second->runningFilter(audio, kHsFrameSize, filterFade::kNoFade);
            }
        }
        if (audioHighband != nullptr) {
            float afterEnergy = 0.0f;
            // std::for_each(std::begin(audio), std::end(audio), [&afterEnergy](float x) { afterEnergy += (x * x); });
            for (size_t i = 0; i < kHsFrameSize; i++) {
                afterEnergy += (audio[i] * audio[i]);
            }
            float gain = afterEnergy / (beforeEnergy + 1e-6f);
            if (gain < 0.9025f) {
                gain = gain > 1e-4f ? sqrtf(gain) : 0.01f;
                //std::transform(std::begin(audioHighband), std::end(audioHighband), std::begin(audioHighband), [gain](float x) { return x * gain; });
                for (size_t i = 0; i < kHsFrameSize; i++) {
                    audioHighband[i] *= gain;
                }
            }
        }
    }

#ifdef HS_DEBUG_RECORD
    if (suppression_condition_) {
        int x = 0;
        if (hasHowling == true) {
            x = 20000;
        } else if (maybeHowling == true) {
            x = 10000;
        }
        for (size_t i = 0; i < kHsFrameSize; i++) {
            hs_record_buffer_[i] = (int16_t)x;
        }
        fwrite(&hs_record_buffer_[0], sizeof(int16_t), kHsFrameSize, suppression_condition_);
    }
#endif // HS_DEBUG_RECORD

#ifdef HS_DEBUG_RECORD
    if (after_hs_) {
        for (size_t i = 0; i < kHsFrameSize; i++) {
            hs_record_buffer_[i] = (int16_t)audio[i];
        }
        fwrite(&hs_record_buffer_[0], sizeof(int16_t), kHsFrameSize, after_hs_);
    }
#endif // HS_DEBUG_RECORD
}

void HowlingSuppressor::Enable(bool enable) {
    enabled_ = enable;
}

}  // namespace webrtc
