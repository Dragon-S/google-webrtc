/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_AUDIO_PROCESSING_HS_NOTCH_FILTER_H_
#define MODULES_AUDIO_PROCESSING_HS_NOTCH_FILTER_H_

#include "hs_common.h"

namespace webrtc {

class NotchFilter {
public:
    explicit NotchFilter(int index, float gain);
    ~NotchFilter();

    void reset();

    void redesign(int index, float gain, float quality);

    void runningFilter(float * audio, int size, filterFade fade);

    void copy(const NotchFilter* reference);

private:
    void design(int index, float gain, float quality);

private:
    float b0_ = 1.0f;
    float b1_ = 0.0f;
    float b2_ = 0.0f;
    float a1_ = 0.0f;
    float a2_ = 0.0f;

    float z1_ = 0.0f;
    float z2_ = 0.0f;
};

} // namespace webrtc

#endif // MODULES_AUDIO_PROCESSING_HS_NOTCH_FILTER_H_
