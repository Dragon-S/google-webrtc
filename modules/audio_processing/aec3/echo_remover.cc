/*
 *  Copyright (c) 2017 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "modules/audio_processing/aec3/echo_remover.h"

#include <math.h>
#include <stddef.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/aec3_fft.h"
#include "modules/audio_processing/aec3/aec_state.h"
#include "modules/audio_processing/aec3/comfort_noise_generator.h"
#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/echo_remover_metrics.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/render_signal_analyzer.h"
#include "modules/audio_processing/aec3/residual_echo_estimator.h"
#include "modules/audio_processing/aec3/subtractor.h"
#include "modules/audio_processing/aec3/subtractor_output.h"
#include "modules/audio_processing/aec3/suppression_filter.h"
#include "modules/audio_processing/aec3/suppression_gain.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

namespace {

// Maximum number of channels for which the capture channel data is stored on
// the stack. If the number of channels are larger than this, they are stored
// using scratch memory that is pre-allocated on the heap. The reason for this
// partitioning is not to waste heap space for handling the more common numbers
// of channels, while at the same time not limiting the support for higher
// numbers of channels by enforcing the capture channel data to be stored on the
// stack using a fixed maximum value.
constexpr size_t kMaxNumChannelsOnStack = 2;

// Chooses the number of channels to store on the heap when that is required due
// to the number of capture channels being larger than the pre-defined number
// of channels to store on the stack.
size_t NumChannelsOnHeap(size_t num_capture_channels) {
  return num_capture_channels > kMaxNumChannelsOnStack ? num_capture_channels
                                                       : 0;
}

void LinearEchoPower(const FftData& E,
                     const FftData& Y,
                     std::array<float, kFftLengthBy2Plus1>* S2) {
  for (size_t k = 0; k < E.re.size(); ++k) {
    (*S2)[k] = (Y.re[k] - E.re[k]) * (Y.re[k] - E.re[k]) +
               (Y.im[k] - E.im[k]) * (Y.im[k] - E.im[k]);
  }
}

// Fades between two input signals using a fix-sized transition.
void SignalTransition(rtc::ArrayView<const float> from,
                      rtc::ArrayView<const float> to,
                      rtc::ArrayView<float> out) {
  if (from == to) {
    RTC_DCHECK_EQ(to.size(), out.size());
    std::copy(to.begin(), to.end(), out.begin());
  } else {
    constexpr size_t kTransitionSize = 30;
    constexpr float kOneByTransitionSizePlusOne = 1.f / (kTransitionSize + 1);

    RTC_DCHECK_EQ(from.size(), to.size());
    RTC_DCHECK_EQ(from.size(), out.size());
    RTC_DCHECK_LE(kTransitionSize, out.size());

    for (size_t k = 0; k < kTransitionSize; ++k) {
      float a = (k + 1) * kOneByTransitionSizePlusOne;
      out[k] = a * to[k] + (1.f - a) * from[k];
    }

    std::copy(to.begin() + kTransitionSize, to.end(),
              out.begin() + kTransitionSize);
  }
}

// Computes a windowed (square root Hanning) padded FFT and updates the related
// memory.
void WindowedPaddedFft(const Aec3Fft& fft,
                       rtc::ArrayView<const float> v,
                       rtc::ArrayView<float> v_old,
                       FftData* V) {
  fft.PaddedFft(v, v_old, Aec3Fft::Window::kSqrtHanning, V);
  std::copy(v.begin(), v.end(), v_old.begin());
}

// Class for removing the echo from the capture signal.
class EchoRemoverImpl final : public EchoRemover {
 public:
  EchoRemoverImpl(const EchoCanceller3Config& config,
                  int sample_rate_hz,
                  size_t num_render_channels,
                  size_t num_capture_channels);
  ~EchoRemoverImpl() override;
  EchoRemoverImpl(const EchoRemoverImpl&) = delete;
  EchoRemoverImpl& operator=(const EchoRemoverImpl&) = delete;

  void GetMetrics(EchoControl::Metrics* metrics) const override;

  // Removes the echo from a block of samples from the capture signal. The
  // supplied render signal is assumed to be pre-aligned with the capture
  // signal.
  void ProcessCapture(EchoPathVariability echo_path_variability,
                      bool capture_signal_saturation,
                      const absl::optional<DelayEstimate>& external_delay,
                      RenderBuffer* render_buffer,
                      Block* linear_output,
                      Block* capture) override;

  // Updates the status on whether echo leakage is detected in the output of the
  // echo remover.
  void UpdateEchoLeakageStatus(bool leakage_detected) override {
    echo_leakage_detected_ = leakage_detected;
  }

  void SetCaptureOutputUsage(bool capture_output_used) override {
    capture_output_used_ = capture_output_used;
  }

 private:
  // Selects which of the coarse and refined linear filter outputs that is most
  // appropriate to pass to the suppressor and forms the linear filter output by
  // smoothly transition between those.
  void FormLinearFilterOutput(const SubtractorOutput& subtractor_output,
                              rtc::ArrayView<float> output);

  bool CorrelateDTD(rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> error,
                  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> capture,
                  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> remote);

  bool FlatnessDTD(rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> error);

  static std::atomic<int> instance_count_;
  const EchoCanceller3Config config_;
  const Aec3Fft fft_;
  std::unique_ptr<ApmDataDumper> data_dumper_;
  const Aec3Optimization optimization_;
  const int sample_rate_hz_;
  const size_t num_render_channels_;
  const size_t num_capture_channels_;
  const bool use_coarse_filter_output_;
  Subtractor subtractor_;
  SuppressionGain suppression_gain_;
  ComfortNoiseGenerator cng_;
  SuppressionFilter suppression_filter_;
  RenderSignalAnalyzer render_signal_analyzer_;
  ResidualEchoEstimator residual_echo_estimator_;
  bool echo_leakage_detected_ = false;
  bool capture_output_used_ = true;
  AecState aec_state_;
  EchoRemoverMetrics metrics_;
  std::vector<std::array<float, kFftLengthBy2>> e_old_;
  std::vector<std::array<float, kFftLengthBy2>> y_old_;
  size_t block_counter_ = 0;
  int gain_change_hangover_ = 0;
  bool refined_filter_output_last_selected_ = true;

  std::vector<std::array<float, kFftLengthBy2>> e_heap_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> Y2_heap_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> E2_heap_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> R2_heap_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> R2_unbounded_heap_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> S2_linear_heap_;
  std::vector<FftData> Y_heap_;
  std::vector<FftData> E_heap_;
  std::vector<FftData> comfort_noise_heap_;
  std::vector<FftData> high_band_comfort_noise_heap_;
  std::vector<SubtractorOutput> subtractor_output_heap_;
  std::vector<std::array<float, 2>> long_term_correlate_;
  std::vector<std::array<float, 2>> long_term_residual_;
  std::vector<float> long_term_flatness_;
  std::vector<float> short_term_flatness_;
};

std::atomic<int> EchoRemoverImpl::instance_count_(0);

EchoRemoverImpl::EchoRemoverImpl(const EchoCanceller3Config& config,
                                 int sample_rate_hz,
                                 size_t num_render_channels,
                                 size_t num_capture_channels)
    : config_(config),
      fft_(),
      data_dumper_(new ApmDataDumper(instance_count_.fetch_add(1) + 1)),
      optimization_(DetectOptimization()),
      sample_rate_hz_(sample_rate_hz),
      num_render_channels_(num_render_channels),
      num_capture_channels_(num_capture_channels),
      use_coarse_filter_output_(
          config_.filter.enable_coarse_filter_output_usage),
      subtractor_(config,
                  num_render_channels_,
                  num_capture_channels_,
                  data_dumper_.get(),
                  optimization_),
      suppression_gain_(config_,
                        optimization_,
                        sample_rate_hz,
                        num_capture_channels),
      cng_(config_, optimization_, num_capture_channels_),
      suppression_filter_(optimization_,
                          sample_rate_hz_,
                          num_capture_channels_),
      render_signal_analyzer_(config_),
      residual_echo_estimator_(config_, num_render_channels),
      aec_state_(config_, num_capture_channels_),
      e_old_(num_capture_channels_, {0.f}),
      y_old_(num_capture_channels_, {0.f}),
      e_heap_(NumChannelsOnHeap(num_capture_channels_), {0.f}),
      Y2_heap_(NumChannelsOnHeap(num_capture_channels_)),
      E2_heap_(NumChannelsOnHeap(num_capture_channels_)),
      R2_heap_(NumChannelsOnHeap(num_capture_channels_)),
      R2_unbounded_heap_(NumChannelsOnHeap(num_capture_channels_)),
      S2_linear_heap_(NumChannelsOnHeap(num_capture_channels_)),
      Y_heap_(NumChannelsOnHeap(num_capture_channels_)),
      E_heap_(NumChannelsOnHeap(num_capture_channels_)),
      comfort_noise_heap_(NumChannelsOnHeap(num_capture_channels_)),
      high_band_comfort_noise_heap_(NumChannelsOnHeap(num_capture_channels_)),
      subtractor_output_heap_(NumChannelsOnHeap(num_capture_channels_)),
      long_term_correlate_(num_capture_channels, {0.0f}),
      long_term_residual_(num_capture_channels, {0.0f}),
      long_term_flatness_(num_capture_channels, -100.0f),
      short_term_flatness_(num_capture_channels, -100.0f) {
  RTC_DCHECK(ValidFullBandRate(sample_rate_hz));
}

EchoRemoverImpl::~EchoRemoverImpl() = default;

void EchoRemoverImpl::GetMetrics(EchoControl::Metrics* metrics) const {
  // Echo return loss (ERL) is inverted to go from gain to attenuation.
  metrics->echo_return_loss = -10.0 * std::log10(aec_state_.ErlTimeDomain());
  metrics->echo_return_loss_enhancement =
      Log2TodB(aec_state_.FullBandErleLog2());
}

void EchoRemoverImpl::ProcessCapture(
    EchoPathVariability echo_path_variability,
    bool capture_signal_saturation,
    const absl::optional<DelayEstimate>& external_delay,
    RenderBuffer* render_buffer,
    Block* linear_output,
    Block* capture) {
  ++block_counter_;
  const Block& x = render_buffer->GetBlock(0);
  Block* y = capture;
  RTC_DCHECK(render_buffer);
  RTC_DCHECK(y);
  RTC_DCHECK_EQ(x.NumBands(), NumBandsForRate(sample_rate_hz_));
  RTC_DCHECK_EQ(y->NumBands(), NumBandsForRate(sample_rate_hz_));
  RTC_DCHECK_EQ(x.NumChannels(), num_render_channels_);
  RTC_DCHECK_EQ(y->NumChannels(), num_capture_channels_);

  // Stack allocated data to use when the number of channels is low.
  std::array<std::array<float, kFftLengthBy2>, kMaxNumChannelsOnStack> e_stack;
  std::array<std::array<float, kFftLengthBy2Plus1>, kMaxNumChannelsOnStack>
      Y2_stack;
  std::array<std::array<float, kFftLengthBy2Plus1>, kMaxNumChannelsOnStack>
      E2_stack;
  std::array<std::array<float, kFftLengthBy2Plus1>, kMaxNumChannelsOnStack>
      R2_stack;
  std::array<std::array<float, kFftLengthBy2Plus1>, kMaxNumChannelsOnStack>
      R2_unbounded_stack;
  std::array<std::array<float, kFftLengthBy2Plus1>, kMaxNumChannelsOnStack>
      S2_linear_stack;
  std::array<FftData, kMaxNumChannelsOnStack> Y_stack;
  std::array<FftData, kMaxNumChannelsOnStack> E_stack;
  std::array<FftData, kMaxNumChannelsOnStack> comfort_noise_stack;
  std::array<FftData, kMaxNumChannelsOnStack> high_band_comfort_noise_stack;
  std::array<SubtractorOutput, kMaxNumChannelsOnStack> subtractor_output_stack;

  rtc::ArrayView<std::array<float, kFftLengthBy2>> e(e_stack.data(),
                                                     num_capture_channels_);
  rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>> Y2(
      Y2_stack.data(), num_capture_channels_);
  rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>> E2(
      E2_stack.data(), num_capture_channels_);
  rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>> R2(
      R2_stack.data(), num_capture_channels_);
  rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>> R2_unbounded(
      R2_unbounded_stack.data(), num_capture_channels_);
  rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>> S2_linear(
      S2_linear_stack.data(), num_capture_channels_);
  rtc::ArrayView<FftData> Y(Y_stack.data(), num_capture_channels_);
  rtc::ArrayView<FftData> E(E_stack.data(), num_capture_channels_);
  rtc::ArrayView<FftData> comfort_noise(comfort_noise_stack.data(),
                                        num_capture_channels_);
  rtc::ArrayView<FftData> high_band_comfort_noise(
      high_band_comfort_noise_stack.data(), num_capture_channels_);
  rtc::ArrayView<SubtractorOutput> subtractor_output(
      subtractor_output_stack.data(), num_capture_channels_);
  if (NumChannelsOnHeap(num_capture_channels_) > 0) {
    // If the stack-allocated space is too small, use the heap for storing the
    // microphone data.
    e = rtc::ArrayView<std::array<float, kFftLengthBy2>>(e_heap_.data(),
                                                         num_capture_channels_);
    Y2 = rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>>(
        Y2_heap_.data(), num_capture_channels_);
    E2 = rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>>(
        E2_heap_.data(), num_capture_channels_);
    R2 = rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>>(
        R2_heap_.data(), num_capture_channels_);
    R2_unbounded = rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>>(
        R2_unbounded_heap_.data(), num_capture_channels_);
    S2_linear = rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>>(
        S2_linear_heap_.data(), num_capture_channels_);
    Y = rtc::ArrayView<FftData>(Y_heap_.data(), num_capture_channels_);
    E = rtc::ArrayView<FftData>(E_heap_.data(), num_capture_channels_);
    comfort_noise = rtc::ArrayView<FftData>(comfort_noise_heap_.data(),
                                            num_capture_channels_);
    high_band_comfort_noise = rtc::ArrayView<FftData>(
        high_band_comfort_noise_heap_.data(), num_capture_channels_);
    subtractor_output = rtc::ArrayView<SubtractorOutput>(
        subtractor_output_heap_.data(), num_capture_channels_);
  }

  data_dumper_->DumpWav("aec3_echo_remover_capture_input",
                        y->View(/*band=*/0, /*channel=*/0), 16000, 1);
  data_dumper_->DumpWav("aec3_echo_remover_render_input",
                        x.View(/*band=*/0, /*channel=*/0), 16000, 1);
  data_dumper_->DumpRaw("aec3_echo_remover_capture_input",
                        y->View(/*band=*/0, /*channel=*/0));
  data_dumper_->DumpRaw("aec3_echo_remover_render_input",
                        x.View(/*band=*/0, /*channel=*/0));

  aec_state_.UpdateCaptureSaturation(capture_signal_saturation);

  if (echo_path_variability.AudioPathChanged()) {
    // Ensure that the gain change is only acted on once per frame.
    if (echo_path_variability.gain_change) {
      if (gain_change_hangover_ == 0) {
        constexpr int kMaxBlocksPerFrame = 3;
        gain_change_hangover_ = kMaxBlocksPerFrame;
        rtc::LoggingSeverity log_level =
            config_.delay.log_warning_on_delay_changes ? rtc::LS_WARNING
                                                       : rtc::LS_VERBOSE;
        RTC_LOG_V(log_level)
            << "Gain change detected at block " << block_counter_;
      } else {
        echo_path_variability.gain_change = false;
      }
    }

    int change_code = (int)echo_path_variability.delay_change;
    if (echo_path_variability.gain_change) {
      change_code = 3;
    }
    RTC_LOG(LS_ERROR)
      << "handle echo path change maybe lead echo leakage with code "
      << change_code;

    subtractor_.HandleEchoPathChange(echo_path_variability);
    aec_state_.HandleEchoPathChange(echo_path_variability);

    if (echo_path_variability.delay_change !=
        EchoPathVariability::DelayAdjustment::kNone) {
      suppression_gain_.SetInitialState(true);
    }
  }
  if (gain_change_hangover_ > 0) {
    --gain_change_hangover_;
  }

  // Analyze the render signal.
  render_signal_analyzer_.Update(*render_buffer,
                                 aec_state_.MinDirectPathFilterDelay());

  // State transition.
  if (aec_state_.TransitionTriggered()) {
    subtractor_.ExitInitialState();
    suppression_gain_.SetInitialState(false);
  }

  // Perform linear echo cancellation.
  subtractor_.Process(*render_buffer, *y, render_signal_analyzer_, aec_state_,
                      subtractor_output);

  // Compute spectra.
  for (size_t ch = 0; ch < num_capture_channels_; ++ch) {
    FormLinearFilterOutput(subtractor_output[ch], e[ch]);
    WindowedPaddedFft(fft_, y->View(/*band=*/0, ch), y_old_[ch], &Y[ch]);
    WindowedPaddedFft(fft_, e[ch], e_old_[ch], &E[ch]);
    LinearEchoPower(E[ch], Y[ch], &S2_linear[ch]);
    Y[ch].Spectrum(optimization_, Y2[ch]);
    E[ch].Spectrum(optimization_, E2[ch]);
  }

  // Optionally return the linear filter output.
  if (linear_output) {
    RTC_DCHECK_GE(1, linear_output->NumBands());
    RTC_DCHECK_EQ(num_capture_channels_, linear_output->NumChannels());
    for (size_t ch = 0; ch < num_capture_channels_; ++ch) {
      std::copy(e[ch].begin(), e[ch].end(),
                linear_output->begin(/*band=*/0, ch));
    }
  }

  // Update the AEC state information.
  aec_state_.Update(external_delay, subtractor_.FilterFrequencyResponses(),
                    subtractor_.FilterImpulseResponses(), *render_buffer, E2,
                    Y2, subtractor_output);

  // Choose the linear output.
  const auto& Y_fft = aec_state_.UseLinearFilterOutput() ? E : Y;

  data_dumper_->DumpWav("aec3_output_linear",
                        y->View(/*band=*/0, /*channel=*/0), 16000, 1);
  data_dumper_->DumpWav("aec3_output_linear2", kBlockSize, &e[0][0], 16000, 1);

  // Estimate the comfort noise.
  cng_.Compute(aec_state_.SaturatedCapture(), Y2, comfort_noise,
               high_band_comfort_noise);

  // Only do the below processing if the output of the audio processing module
  // is used.
  std::array<float, kFftLengthBy2Plus1> G;
  if (capture_output_used_) {
    // correlate based DTD
    bool DTD = false;
    if (aec_state_.UsableLinearEstimate()) {
      bool condition1 = CorrelateDTD(E2, Y2, render_buffer->Spectrum(0));
      bool condition2 = FlatnessDTD(E2);
      DTD = condition1 && condition2;
    }

    // Estimate the residual echo power.
    residual_echo_estimator_.Estimate(aec_state_, *render_buffer, S2_linear, Y2,
                                      suppression_gain_.IsDominantNearend(), R2,
                                      R2_unbounded);

    // Suppressor nearend estimate.
    if (aec_state_.UsableLinearEstimate()) {
      // E2 is bound by Y2.
      for (size_t ch = 0; ch < num_capture_channels_; ++ch) {
        std::transform(E2[ch].begin(), E2[ch].end(), Y2[ch].begin(),
                       E2[ch].begin(),
                       [](float a, float b) { return std::min(a, b); });
      }
    }
    const auto& nearend_spectrum = aec_state_.UsableLinearEstimate() ? E2 : Y2;

    // Suppressor echo estimate.
    const auto& echo_spectrum =
        aec_state_.UsableLinearEstimate() ? S2_linear : R2;

    // Determine if the suppressor should assume clock drift.
    const bool clock_drift = config_.echo_removal_control.has_clock_drift ||
                             echo_path_variability.clock_drift;

    // Compute preferred gains.
    float high_bands_gain;
    suppression_gain_.GetGain(nearend_spectrum, echo_spectrum, R2, R2_unbounded,
                              cng_.NoiseSpectrum(), render_signal_analyzer_,
                              aec_state_, x, clock_drift, &high_bands_gain, &G, DTD);

    suppression_filter_.ApplyGain(comfort_noise, high_band_comfort_noise, G,
                                  high_bands_gain, Y_fft, y);

  } else {
    G.fill(0.f);
  }

  // Update the metrics.
  metrics_.Update(aec_state_, cng_.NoiseSpectrum()[0], G);

  // Debug outputs for the purpose of development and analysis.
  data_dumper_->DumpWav("aec3_echo_estimate", kBlockSize,
                        &subtractor_output[0].s_refined[0], 16000, 1);
  data_dumper_->DumpRaw("aec3_output", y->View(/*band=*/0, /*channel=*/0));
  data_dumper_->DumpRaw("aec3_narrow_render",
                        render_signal_analyzer_.NarrowPeakBand() ? 1 : 0);
  data_dumper_->DumpRaw("aec3_N2", cng_.NoiseSpectrum()[0]);
  data_dumper_->DumpRaw("aec3_suppressor_gain", G);
  data_dumper_->DumpWav("aec3_output", y->View(/*band=*/0, /*channel=*/0),
                        16000, 1);
  data_dumper_->DumpRaw("aec3_using_subtractor_output[0]",
                        aec_state_.UseLinearFilterOutput() ? 1 : 0);
  data_dumper_->DumpRaw("aec3_E2", E2[0]);
  data_dumper_->DumpRaw("aec3_S2_linear", S2_linear[0]);
  data_dumper_->DumpRaw("aec3_Y2", Y2[0]);
  data_dumper_->DumpRaw(
      "aec3_X2", render_buffer->Spectrum(
                     aec_state_.MinDirectPathFilterDelay())[/*channel=*/0]);
  data_dumper_->DumpRaw("aec3_R2", R2[0]);
  data_dumper_->DumpRaw("aec3_filter_delay",
                        aec_state_.MinDirectPathFilterDelay());
  data_dumper_->DumpRaw("aec3_capture_saturation",
                        aec_state_.SaturatedCapture() ? 1 : 0);
}

void EchoRemoverImpl::FormLinearFilterOutput(
    const SubtractorOutput& subtractor_output,
    rtc::ArrayView<float> output) {
  RTC_DCHECK_EQ(subtractor_output.e_refined.size(), output.size());
  RTC_DCHECK_EQ(subtractor_output.e_coarse.size(), output.size());
  bool use_refined_output = true;
  if (use_coarse_filter_output_) {
    // As the output of the refined adaptive filter generally should be better
    // than the coarse filter output, add a margin and threshold for when
    // choosing the coarse filter output.
    if (subtractor_output.e2_coarse < 0.9f * subtractor_output.e2_refined &&
        subtractor_output.y2 > 30.f * 30.f * kBlockSize &&
        (subtractor_output.s2_refined > 60.f * 60.f * kBlockSize ||
         subtractor_output.s2_coarse > 60.f * 60.f * kBlockSize)) {
      use_refined_output = false;
    } else {
      // If the refined filter is diverged, choose the filter output that has
      // the lowest power.
      if (subtractor_output.e2_coarse < subtractor_output.e2_refined &&
          subtractor_output.y2 < subtractor_output.e2_refined) {
        use_refined_output = false;
      }
    }
  }

  SignalTransition(refined_filter_output_last_selected_
                       ? subtractor_output.e_refined
                       : subtractor_output.e_coarse,
                   use_refined_output ? subtractor_output.e_refined
                                      : subtractor_output.e_coarse,
                   output);
  refined_filter_output_last_selected_ = use_refined_output;
}

// a correlate based method for DTD
// return TRUE if nearend voice activate detected when remote voice activated
bool EchoRemoverImpl::CorrelateDTD(rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> error,
                  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> capture,
                  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> remote) {
  // e-x correlate
  float feature1 = -100.0f;
  for (size_t i = 0; i < error.size(); i++) {
    float EX = 0.0f, EE = 0.0f, XX = 0.0f;
    for (size_t j = 0; j < kFftLengthBy2Plus1 / 2; j++) {
      EE += error[i][j] * error[i][j];
      XX += remote[i][j] * remote[i][j];
      EX += error[i][j] * remote[i][j];
    }
    float temp = EX / sqrtf(XX * EE + 1e-6f);
    float correlateLow = 0.999f * long_term_correlate_[i][0] + 0.001f * temp;
    long_term_correlate_[i][0] = correlateLow < 1.0f ? correlateLow : 1.0f;

    EX = EE = XX = 0.0f;
    for (size_t j = kFftLengthBy2Plus1 / 2; j < kFftLengthBy2Plus1 - 1; j++) {
      EE += error[i][j] * error[i][j];
      XX += remote[i][j] * remote[i][j];
      EX += error[i][j] * remote[i][j];
    }
    temp = EX / sqrtf(XX * EE + 1e-6f);
    float correlateHigh = 0.999f * long_term_correlate_[i][1] + 0.001f * temp;
    long_term_correlate_[i][1] = correlateHigh < 1.0f ? correlateHigh : 1.0f;

    temp = long_term_correlate_[i][1] / (long_term_correlate_[i][0] + 1e-6f);
    feature1 = feature1 < temp ? temp : feature1;
  }

  // e-y residual
  float feature2 = -100.0f;
  for (size_t i = 0; i < error.size(); i++) {
    float E = 0.0f, Y = 0.0f;
    for (size_t j = 0; j < kFftLengthBy2Plus1 / 2; j++) {
      E += error[i][j];
      Y += capture[i][j];
    }
    float temp = E / (Y + 1e-6f);
    float residualLow = 0.999f * long_term_residual_[i][0] + 0.001f * temp;
    long_term_residual_[i][0] = residualLow < 1.0f ? residualLow : 1.0f;

    E = Y = 0.0f;
    for (size_t j = kFftLengthBy2Plus1 / 2; j < kFftLengthBy2Plus1 - 1; j++) {
      E += error[i][j];
      Y += capture[i][j];
    }
    temp = E / (Y + 1e-6f);
    float residualHigh = 0.999f * long_term_residual_[i][1] + 0.001f * temp;
    long_term_residual_[i][1] = residualHigh < 1.0f ? residualHigh : 1.0f;

    temp = long_term_residual_[i][1] / (long_term_residual_[i][0] + 1e-6f);
    feature2 = feature2 < temp ? temp : feature2;
  }

  return feature1 > 0.95f && feature2 > 0.85f;
}

// a flatness based method for DTD
bool EchoRemoverImpl::FlatnessDTD(rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> error) {
  // e flatness
  float feature = -100.0f;
  for (size_t i = 0; i < error.size(); i++) {
    float temp1 = 0.0f, temp2 = 0.0f;
    for (size_t j = 0; j < kFftLengthBy2Plus1 / 2; j++) {
      temp1 += logf(error[i][j] + 1e-6f);
      temp2 += error[i][j];
    }
    float geomitricMeanLow = expf(temp1 / (float)(kFftLengthBy2Plus1 / 2.0f));
    float arithmeticMeanLow = temp2 / (float)(kFftLengthBy2Plus1 / 2.0f);
    float flatnessLow = geomitricMeanLow / (arithmeticMeanLow + 1e-6f);

    temp1 = temp2 = 0.0f;
    for (size_t j = kFftLengthBy2Plus1 / 2; j < kFftLengthBy2Plus1 - 1; j++) {
      temp1 += logf(error[i][j] + 1e-6f);
      temp2 += error[i][j];
    }
    float geomitricMeanHigh = expf(temp1 / (float)(kFftLengthBy2Plus1 / 2.0f));
    float arithmeticMeanHigh = temp2 / (float)(kFftLengthBy2Plus1 / 2.0f);
    float flatnessHigh = geomitricMeanHigh / (arithmeticMeanHigh + 1e-6f);

    float temp = flatnessHigh / (flatnessLow + 1e-6f);
    if (long_term_flatness_[i] < 0.0f) {
      long_term_flatness_[i] = temp;
    } else {
      long_term_flatness_[i] = 0.9999f * long_term_flatness_[i] + 0.0001f * temp;
    }
    if (short_term_flatness_[i] < 0.0f) {
      short_term_flatness_[i] = temp;
    } else {
      short_term_flatness_[i] = 0.9f * short_term_flatness_[i] + 0.1f * temp;
    }

    temp = short_term_flatness_[i] / (long_term_flatness_[i] + 1e-6f);
    feature = feature < temp ? temp : feature;
  }

  return feature > 1.2f;
}

}  // namespace

EchoRemover* EchoRemover::Create(const EchoCanceller3Config& config,
                                 int sample_rate_hz,
                                 size_t num_render_channels,
                                 size_t num_capture_channels) {
  return new EchoRemoverImpl(config, sample_rate_hz, num_render_channels,
                             num_capture_channels);
}

}  // namespace webrtc
