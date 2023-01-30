/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "notch_filter.h"
#include "hs_common.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

namespace webrtc {

NotchFilter::NotchFilter(int index, float gain) {
    design(index, gain, 10.0f);
}

NotchFilter::~NotchFilter() {}

void NotchFilter::reset() {
    z1_ = 0.0f;
    z2_ = 0.0f;
}

void NotchFilter::redesign(int index, float gain, float quality) {
    design(index, gain, quality);
    reset();
}

void NotchFilter::runningFilter(float * audio, int size, filterFade fade) {
    if ((audio == nullptr) || (size == 0) || (size > 480)) {
        return;
    }

    float z0 = 0.0f;
    for (size_t i = 0; i < (size_t)size; i++) {
        z0 = audio[i] - a1_ * z1_ - a2_ * z2_;
        if (fade == filterFade::kFadeOut) {
            float fadeValue = (float)(size - i) / (float)size;
            audio[i] = fadeValue * (b0_ * z0 + b1_ * z1_ + b2_ * z2_) + (1.0f - fadeValue) * audio[i];
        } else if (fade == filterFade::kFadeIn) {
            float fadeValue = 0.0f;
            if (i > (size_t)size / 2) {
                fadeValue = (float)(i - size / 2) / (float)(size - size / 2);
            }
            audio[i] = fadeValue * (b0_ * z0 + b1_ * z1_ + b2_ * z2_) + (1.0f - fadeValue) * audio[i];            
        } else {
            audio[i] = b0_ * z0 + b1_ * z1_ + b2_ * z2_;
        }
        z2_ = z1_;
        z1_ = z0;
    }
}

void NotchFilter::copy(const NotchFilter* reference) {
    z2_ = reference->z2_;
    z1_ = reference->z1_;

    a2_ = reference->a2_;
    a1_ = reference->a1_;

    b2_ = reference->b2_;
    b1_ = reference->b1_;
    b0_ = reference->b0_;
}

void NotchFilter::design(int index, float gain, float quality) {
    if (index < 1) {
        index = 1;
    } else if ((size_t)index >= kHsFftSizeBy2) {
        index = kHsFftSizeBy2 - 1;
    }
    float cw = (index * 62.5f) * acosf(-1.0f) / 8000.0f;
    float bilinear = tanf(cw / 2.0f);
    float bilinear2 = powf(bilinear, 2.0f);
    
    float den = 1.0f + bilinear / quality + bilinear2;
    if (den != 0.0f) {
        b0_ = (1.0f + gain * bilinear / quality + bilinear2) / den;
        b1_ = 2.0f * (bilinear2 - 1.0f) / den;
        b2_ = (1.0f - gain * bilinear / quality + bilinear2) / den;
        a1_ = 2.0f * (bilinear2 - 1.0f) / den;
        a2_ = (1.0f - bilinear / quality + bilinear2) / den;
    }
}

} // namespace webrtc