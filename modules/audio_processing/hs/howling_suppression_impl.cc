/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "howling_suppression_impl.h"

#include "modules/audio_processing/audio_buffer.h"
#include "howling_suppressor.h"


namespace webrtc {

HowlingSuppressionImpl::HowlingSuppressionImpl() {}

HowlingSuppressionImpl::~HowlingSuppressionImpl() {}

void HowlingSuppressionImpl::Initialize(size_t channels, int sample_rate_hz) {
  //RTC_DCHECK(sample_rate_hz == 48000);
  channels_ = channels;
  sample_rate_hz_ = sample_rate_hz;
  
  std::vector<std::unique_ptr<HowlingSuppressor> > new_howlingSuppressor;
  if (enabled_) {
    new_howlingSuppressor.resize(channels);
    for (size_t i = 0; i < channels; i++) {
      new_howlingSuppressor[i].reset(new HowlingSuppressor());
      new_howlingSuppressor[i]->Enable(enabled_);
    }
  }
  howlingSuppressor_.swap(new_howlingSuppressor);
}

int HowlingSuppressionImpl::ProcessCaptureAudio(AudioBuffer* audio) {
  RTC_DCHECK(audio);
  if (!enabled_) {
    return AudioProcessing::kNotEnabledError;
  }

  for (size_t i = 0; i < howlingSuppressor_.size(); i++) {
      if (audio->num_bands() <= 1) {
        howlingSuppressor_[i]->Process(audio->split_bands_f(i)[0], nullptr);
      } else {
        howlingSuppressor_[i]->Process(audio->split_bands_f(i)[0], audio->split_bands_f(i)[1]);
      }
  }
  return AudioProcessing::kNoError;
}

int HowlingSuppressionImpl::Enable(bool enable) {
  if (enabled_ != enable) {
    enabled_ = enable;
    Initialize(channels_, sample_rate_hz_);
  }
  return AudioProcessing::kNoError;
}

bool HowlingSuppressionImpl::is_enabled() const {
  return enabled_;
}


}  // namespace webrtc
