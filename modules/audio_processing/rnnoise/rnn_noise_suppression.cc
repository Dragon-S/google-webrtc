/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "modules/audio_processing/rnnoise/rnn_noise_suppression.h"

#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/rnnoise/rnnoise.h"


namespace webrtc {
class RnnNoiseSuppressionImpl::RnnSuppressor {
 public:
  // explicit constructor must different with implicit one (just take a argment)
  explicit RnnSuppressor(int sample_rate_hz) {
    // sample_rate_hz = sample_rate_hz;
    state_ = (DenoiseState*)rnnoise_create();
    RTC_CHECK(state_);
    int error = rnnoise_init(state_);
    RTC_DCHECK_EQ(0, error);
  }
  ~RnnSuppressor() { rnnoise_destroy(state_); }
  DenoiseState* state() { return state_; }

 private:
  DenoiseState* state_ = nullptr;
};

RnnNoiseSuppressionImpl::RnnNoiseSuppressionImpl() {}

RnnNoiseSuppressionImpl::~RnnNoiseSuppressionImpl() {}

void RnnNoiseSuppressionImpl::Initialize(size_t channels, int sample_rate_hz) {
  RTC_DCHECK(sample_rate_hz == 16000);
  channels_ = channels;
  sample_rate_hz_ = sample_rate_hz;
  std::vector<std::unique_ptr<RnnSuppressor>> new_suppressors;
  if (enabled_) {
    new_suppressors.resize(channels);
    for (size_t i = 0; i < channels; i++) {
      new_suppressors[i].reset(new RnnSuppressor(sample_rate_hz));
    }
  }
  suppressors_.swap(new_suppressors);
}

void RnnNoiseSuppressionImpl::ProcessCaptureAudio(AudioBuffer* audio) {
  RTC_DCHECK(audio);
  if (!enabled_) {
    return;
  }

  RTC_DCHECK_GE(160, audio->num_frames_per_band());
  RTC_DCHECK_EQ(suppressors_.size(), audio->num_channels());
  for (size_t i = 0; i < suppressors_.size(); i++) {
      float enery = 1.0f;
      rnnoise_process_frame(suppressors_[i]->state(), audio->split_bands_f(i)[0], audio->split_bands_const_f(i)[0], &enery);
      for (size_t j = 1; j < audio->num_bands(); j++) {
        memset(audio->split_bands_f(i)[j], 0, audio->num_frames_per_band() * sizeof(audio->split_bands_f(i)[j][0]));
      }
  }
}

int RnnNoiseSuppressionImpl::Enable(bool enable) {
  if (enabled_ != enable) {
    enabled_ = enable;
    Initialize(channels_, sample_rate_hz_);
  }
  return AudioProcessing::kNoError;
}

bool RnnNoiseSuppressionImpl::is_enabled() const {
  return enabled_;
}

float RnnNoiseSuppressionImpl::speech_probability() const {
//   rtc::CritScope cs(crit_);
  return 1.0f;
}


}  // namespace webrtc
