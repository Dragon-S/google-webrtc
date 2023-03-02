#include "personal_ns.h"
#include "mel_feature.h"
#include "embedder.h"
#include "pns_fft.h"
#include "voice_filter.h"

#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <exception>
#include <cassert>
#if defined(WEBRTC_WIN)
#include <shlobj.h>
#endif
#include <string>

#include "modules/third_party/helper/helper.h"
#include "rtc_base/logging.h"

namespace pns
{
#if defined(WEBRTC_WIN)
static std::wstring getXuanXingMeetPath() {
    PWSTR appDataPath = nullptr;
    std::wstring resultPath;
    if (SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &appDataPath) == S_OK) {
        CoTaskMemFree(appDataPath);
        resultPath = std::wstring(appDataPath) + L"\\XuanXingMeet";
    } else {
        CoTaskMemFree(appDataPath);
        RTC_LOG(LS_ERROR) << "Failed to get appdata path";
    }

    return resultPath;
}

static std::string getStringPathFromWString(const std::wstring fileName) {
    std::wstring wsPath = getXuanXingMeetPath() + L"\\" + fileName;
    if (wsPath == L"") {
        RTC_LOG(LS_ERROR) << "path is null";
        return "";
    }
    const wchar_t* pwsPath = wsPath.c_str();
    char cPath[MAX_PATH];
    wcstombs(cPath, pwsPath, MAX_PATH);
    return std::string(cPath);
}
#endif

static std::string getDVectorBinFilePath() {
    std::string dVectorBinFilePath = "";
#ifdef WEBRTC_WIN
    dVectorBinFilePath = getStringPathFromWString(L"dVector.bin");
#elif WEBRTC_MAC
    dVectorBinFilePath = helper::GetXuanXingMeetDir() + "/dVector.bin";
#endif
    return dVectorBinFilePath;
}

static auto getVoiceFilterModelFilePath() {
#ifdef WEBRTC_WIN
    std::wstring voicefilterModelFilePathWS = getXuanXingMeetPath() + L"\\voicefilterModel.onnx";
    return voicefilterModelFilePathWS;
#elif WEBRTC_MAC
    std::string voicefilterModelFilePathS = helper::GetXuanXingMeetDir() + "/voicefilterModel.onnx";
    return voicefilterModelFilePathS;
#endif
}

static auto getEmbedderModelFilePath() {
#ifdef WEBRTC_WIN
    std::wstring embedderModelFilePathWS = getXuanXingMeetPath() + L"\\embedderModel.onnx";
    return embedderModelFilePathWS;
#elif WEBRTC_MAC
    std::string embedderModelFilePathS = helper::GetXuanXingMeetDir() + "/embedderModel.onnx";
    return embedderModelFilePathS;
#endif
}

static std::string getPCMFilePath() {
    std::string pcmFilePath = "";
#ifdef WEBRTC_WIN
    pcmFilePath = getStringPathFromWString(L"uttrance.pcm");
#elif WEBRTC_MAC
    pcmFilePath = helper::GetXuanXingMeetDir() + "/uttrance.pcm";
#endif
    return pcmFilePath;
}

PersonalNs* PersonalNs::createPns() {
    return dynamic_cast<PersonalNs *>(new PersonalNsImpl());
}

PersonalNsImpl::PersonalNsImpl() : startPhase_(0), writePosition_(0), readPosition_(1), mode_(PersonalNsMode::slight), voicefilterTurnOn_(false), smoothedRms_(0.5f), slightHurtCount_(0), seriousHurtCount_(0) {
    dVector_.clear();
    realLatency_.resize(5);
    imagLatency_.resize(5);
    ampLatency_.resize(5);
    recvBuffer_.resize(reciveSize_, 0.0f);
    synBuffer_.resize(windowSize_, 0.0f);
    synWindow_.resize(windowSize_, 0.2f);
    pnsFft_.reset(new PnsFft(fftSize_));
    melFeature_.reset(new MelFeature(melFrameSize_, melFftSize_, melNumber_, melLowsetEnergy_));
    embedder_.reset();
    auto voiceFileterModelPath = getVoiceFilterModelFilePath();
    const ORTCHAR_T* voicefilterModel = voiceFileterModelPath.c_str();
    voiceFilter_.reset(new VoiceFilter(voicefilterModel, specturmSize_, embedderDim_, hiddenNumber_));
    noisyRms_.resize(5, 0.0f);

    for (size_t i = 0; i < windowSize_ / 2; i++) {
        float v = (float)i / (float)(windowSize_ / 2 - 1);
        synWindow_[i] = v;
        synWindow_[windowSize_ - 1 - i] = v;
    }

    for (size_t i = 0; i < 5; i++) {
        realLatency_[i].reset(new std::vector<float>(fftSize_, 0.0f));
        imagLatency_[i].reset(new std::vector<float>(fftSize_, 0.0f));
        ampLatency_[i].reset(new std::vector<float>(specturmSize_, 0.0f));
    }
}

PersonalNsImpl::~PersonalNsImpl() {
    dVector_.clear();
    realLatency_.clear();
    imagLatency_.clear();
    ampLatency_.clear();
    recvBuffer_.clear();
    synBuffer_.clear();
    synWindow_.clear();
    melFeature_.reset();
    embedder_.reset();
    voiceFilter_.reset();
    noisyRms_.clear();
}

int PersonalNsImpl::extractEmbedder() {
    // if dvector had extracted, just return
    std::string dVectorBinPath = getDVectorBinFilePath();
    if (dVectorBinPath == "") {
        RTC_LOG(LS_ERROR) << "dVectorBinPath is null";
        return -1;
    }

    std::ifstream inFile(std::ifstream(dVectorBinPath, std::ios_base::binary | std::ios_base::in));
    if (inFile.good()) {
        return 0;
    }

    // uttrance speech and d-vector extract
    size_t length = 0;
    size_t sampleNumber = 0;
    int16_t* uttranceSpeechBuf = nullptr;

    std::string pcmFilePath = getPCMFilePath();
    if (pcmFilePath == "") {
        RTC_LOG(LS_ERROR) << "pcmFilePath is null";
        return -1;
    }
    std::ifstream uttranceSpeechSteam(pcmFilePath, std::ios_base::binary | std::ios_base::in);
    if (!uttranceSpeechSteam.good()) {
        RTC_LOG(LS_ERROR) << "uttranceSpeechSteam is not good";
        return -1;
    }
    uttranceSpeechSteam.seekg(0, std::ios_base::end);
    length = uttranceSpeechSteam.tellg();
    assert(length % sizeof(int16_t) == 0);

    sampleNumber = length / sizeof(int16_t);
    uttranceSpeechBuf = new int16_t[sampleNumber];
    uttranceSpeechSteam.seekg(0, std::ios_base::beg);
    uttranceSpeechSteam.read((char *)uttranceSpeechBuf, length);
    uttranceSpeechSteam.close();

    // convert from i16 to f32
    float * uttranceSpeech = new float[sampleNumber];
    for (size_t i = 0; i < sampleNumber; i++) {
        uttranceSpeech[i] = uttranceSpeechBuf[i] / 32768.0f;
    }
    delete []uttranceSpeechBuf;

    if (melFeature_) {
        assert(length > melFrameSize_ * 10);
        std::vector<std::vector<float> > mels = melFeature_->getFeatures(uttranceSpeech, sampleNumber);
        if ((mels.size() <= 0) || (mels.size() > 640)) {
            delete []uttranceSpeech;
            return -1;
        }

        auto embedderModelFilePath = getEmbedderModelFilePath();
        const ORTCHAR_T* embedderModel = embedderModelFilePath.c_str();
        embedder_.reset(new Embedder(embedderModel, melNumber_, embedderDim_));
        std::vector<float> dVector = embedder_->process(mels);
        delete []uttranceSpeech;
        return saveEmbedder(dVector);
    } else {
        delete []uttranceSpeech;
        return -1;
    }
}

int PersonalNsImpl::processFrame(float* frame) {
    if (dVector_.size() == 0) {
        if (loadEmbedder()) {
            return -1;
        }
        assert(dVector_.size() == embedderDim_);
        voiceFilter_->setDVector(dVector_);
    }

    assert(frame != nullptr);

    // recive input data
    std::copy(recvBuffer_.begin() + frameSize_, recvBuffer_.end(), recvBuffer_.begin());
    for (size_t i = 0; i < frameSize_; i++) {
        recvBuffer_[reciveSize_ - frameSize_ + i] = frame[i];
    }

    // wait for fill recive buffer
    if (startPhase_++ < 2) {
        return 0;
    }

    // get write banks
    auto imagW = imagLatency_[writePosition_];
    auto realW = realLatency_[writePosition_];
    auto ampW = ampLatency_[writePosition_];

    // ready to processing
    imagW->resize(fftSize_, 0.0f);
    realW->resize(fftSize_, 0.0f);
    std::copy(recvBuffer_.begin(), recvBuffer_.begin() + windowSize_, realW->begin());
    // frame rms
    if (mode_ == PersonalNsMode::slight) {
        float energy = 0.0f;
        std::for_each(realW->begin(), realW->begin() + windowSize_, [&energy](float a) { energy += a * a; });
        smoothedRms_ = 0.9f * smoothedRms_ + 0.1f * sqrtf(energy / (float)windowSize_);
        noisyRms_[writePosition_] = smoothedRms_;
    }

    // to frequency domain and get specturm
    pnsFft_->Fft(realW->data(), imagW->data());
    // transform to normalize specturm
    std::transform(realW->begin(), realW->begin() + specturmSize_, imagW->begin(), ampW->begin(), [](float a, float b) { float x = sqrtf(a * a + b * b); return x > 3.2e-3f ? x : 3.2e-3f; });
    std::transform(realW->begin(), realW->begin() + specturmSize_, ampW->begin(), realW->begin(), [](float a, float b) { return a / b; });
    std::transform(imagW->begin(), imagW->begin() + specturmSize_, ampW->begin(), imagW->begin(), [](float a, float b) { return a / b; });
    std::transform(ampW->begin(), ampW->end(), ampW->begin(), [](float a) { float x = log10f(a); return 0.2f * x + 0.5f; });

    // send specturm and dvector into voicefilter
    std::vector<float> mask = voiceFilter_->process(ampW);

    // get read banks
    auto imagR = imagLatency_[readPosition_];
    auto realR = realLatency_[readPosition_];
    auto ampR = ampLatency_[readPosition_];

    // do filter (option on read banks)
    if ((mode_ == PersonalNsMode::normal) || ((mode_ == PersonalNsMode::slight) && (voicefilterTurnOn_))) {
        std::transform(mask.begin(), mask.end(), ampR->begin(), ampR->begin(), [](float a, float b) { return a * b; });
    }
    // tranform back to specturm
    std::transform(ampR->begin(), ampR->end(), ampR->begin(), [](float a) { return powf(10.0f, 5.0f * a - 2.5f); });

    // segment enhance and synthesis
    std::transform(realR->begin(), realR->begin() + specturmSize_, ampR->begin(), realR->begin(), [](float a, float b) { return a * b; });
    std::transform(imagR->begin(), imagR->begin() + specturmSize_, ampR->begin(), imagR->begin(), [](float a, float b) { return a * b; });
    for (size_t i = specturmSize_; i < fftSize_; i++) {
        realR->data()[i] = realR->data()[fftSize_ - i];
        imagR->data()[i] = -imagR->data()[fftSize_ - i];
    }
    pnsFft_->Ifft(realR->data(), imagR->data());
    
    // review the status
    if (mode_ == PersonalNsMode::slight) {
        float energy = 0.0f;
        std::for_each(realR->begin(), realR->begin() + windowSize_, [&energy](float a) { energy += a * a; });
        float rms = sqrtf(energy / (float)windowSize_);
        if (noisyRms_[readPosition_] > 0.0630956f) {
            if (rms / noisyRms_[readPosition_] <= 0.8f) {
                slightHurtCount_ = 0;
                seriousHurtCount_++;
                if (seriousHurtCount_ >= 5) {
                    voicefilterTurnOn_ = false;
                }
            } else {
                slightHurtCount_++;
                seriousHurtCount_ = 0;
                if (slightHurtCount_ >= 100) {
                    voicefilterTurnOn_ = true;
                }
            }
        } else if (noisyRms_[readPosition_] < 0.0251188f) {
            voicefilterTurnOn_ = true;
            slightHurtCount_ = 0;
            seriousHurtCount_ = 0;
        }
    }

    // synthesis 
    std::transform(synWindow_.begin(), synWindow_.end(), realR->begin(), realR->begin(), [](float a, float b) { return a * b; });
    std::transform(synBuffer_.begin(), synBuffer_.end(), realR->begin(), synBuffer_.begin(), [](float a, float b) { return a + b; });

    // output and over processing
    for (size_t i = 0; i < frameSize_; i++) {
        frame[i] = synBuffer_[i];
    }
    std::copy(synBuffer_.begin() + frameSize_, synBuffer_.end(), synBuffer_.begin());
    for (size_t i = 0; i < frameSize_; i++) {
        synBuffer_[windowSize_ - frameSize_ + i] = 0.0f;
    }

    //
    readPosition_ = (readPosition_ + 1) % 5;
    writePosition_ = (writePosition_ + 1) % 5;

    return 0;
}

int PersonalNsImpl::saveEmbedder(std::vector<float> dVector) {
    std::string dVectorBinPath = getDVectorBinFilePath();
    if (dVectorBinPath == "") {
        RTC_LOG(LS_ERROR) << "dVectorBinPath is null";
    }
    std::ofstream outFile(std::ofstream(dVectorBinPath, std::ios_base::binary | std::ios_base::out));
    outFile.write((const char *)dVector.data(), sizeof(float) * embedderDim_);
    outFile.flush();
    outFile.close();

    return 0;
}

int PersonalNsImpl::loadEmbedder() {
    dVector_.resize(embedderDim_, 0.0f);
    std::string dVectorBinPath = getDVectorBinFilePath();
    if (dVectorBinPath == "") {
        RTC_LOG(LS_ERROR) << "dVectorBinPath is null";
    }
    std::ifstream inFile(std::ifstream(dVectorBinPath, std::ios_base::binary | std::ios_base::in));
    if (!inFile.good()) {
        return -1;
    }
    inFile.read((char *)dVector_.data(), sizeof(float) * embedderDim_);

    return 0;
}

} // namespace pns
