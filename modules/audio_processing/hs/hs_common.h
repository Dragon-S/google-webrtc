/*
 *  Copyright (c) 2019 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_AUDIO_PROCESSING_HS_HS_COMMON_H_
#define MODULES_AUDIO_PROCESSING_HS_HS_COMMON_H_

#include <cstddef>
#include <cmath>

namespace webrtc {
const size_t kHsFrameSize = 160;

const float kMPi = 3.141592653589793f;
const float kConstBandwidth = 100.0f;

const size_t kHsFftSize = 256;
const size_t kHsFftSizeBy2 = kHsFftSize / 2;
const size_t kHsFftSizeBy2Plus1 = kHsFftSize / 2 + 1;
const size_t kHsFftExp = 8;

const float kSampleRateHz = 16000.0f;
const float kHzPerBin = kSampleRateHz / kHsFftSize;

const size_t kAutocorrelationTimes = 9;
const size_t kAutoCorrelationLength = kHsFrameSize - kAutocorrelationTimes;

const float kHowlingJudgeThreshold1 = 1.585f;
const float kHowlingJudgeThreshold2 = 3.0f;
const float kHowlingJudgeThreshold3 = 5;
const float kHowlingJudgeThreshold4 = 1;
const float kHowlingJudgeThreshold5 = 0.90f;
const float kHowlingJudgeThreshold6 = 0.45f;
const float kHowlingJudgeThreshold7 = 2.3219f;

const float kAttackFactor = 0.2f;
const float kReleaseFactor = 0.9f;

const size_t kHsLpcOrder = 18;

const float kHsPerceptronModel[kHsFftSizeBy2] = {
    -7.7272, 3.7362, 11.9047, -9.1530, -3.8769, -4.4368, -2.6529, 1.7809, 
    -7.1421, 4.8420, 2.6640, -7.3017, -9.3746, 9.8005, -0.5949, 2.0700, 
    3.0269, 5.9048, -6.7939, -0.1881, 0.1747, -6.2445, 8.3340, -7.9999, 
    7.5885, -7.3039, 0.0549, -6.3128, -1.5254, -0.0589, 3.1529, -1.9930, 
    -11.4023, 9.7257, 6.8065, 2.3757, -5.8643, 7.1431, -0.5983, -0.3207, 
    5.5845, 2.4658, 0.7418, -9.5776, -3.0657, 7.2486, -0.2034, -1.7016, 
    -5.1184, 0.9796, -5.3344, -0.1705, -9.0601, -1.4107, -0.9054, -5.7047, 
    3.2476, 2.8252, -7.9306, -5.3104, 2.5256, -5.1454, -5.2187, -3.1036, 
    15.4931, -3.7936, 3.8484, -3.4083, -12.4609, -2.7647, 2.9080, 0.4166, 
    0.3740, 6.7428, -2.6398, -2.8110, -4.1262, -2.9968, -0.4834, 7.0881, 
    -2.5261, 2.9803, -1.2297, 5.4016, -4.6630, -0.0614, 1.1132, -0.3115, 
    -0.7948, -0.3123, -7.2833, -5.0574, -8.2364, 3.5211, 7.1396, -7.5670, 
    13.9414, 2.0945, 6.7066, 1.2705, 3.4687, 2.6688, 11.7838, -6.6963, 
    8.6017, 9.8097, 6.6096, -3.4775, 6.3845, 4.7371, -1.8891, -1.4825, 
    -12.7654, -7.6494, 17.7135, -24.2523, 0.0686, -40.9177, 52.3325, -10.3718, 
    8.2634, -22.5329, 3.8653, 29.7044, -14.1116, -62.7031, -7.7792, 35.8188
};

const size_t kMaxNotchFilter = 8;

enum filterFade {
    kNoFade = 0,
    kFadeIn,
    kFadeOut,
};
const size_t kFadeInMark = 10000;
const size_t kFadeOutMark = 20000;

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_HS_HS_COMMON_H_
