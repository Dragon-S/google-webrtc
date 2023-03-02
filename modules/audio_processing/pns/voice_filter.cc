#include "voice_filter.h"

#include <array>
#include <cmath>
#include <algorithm>
#if defined(WEBRTC_WIN)
#include <windows.h>
#include <windowsx.h>
#endif

#if defined(WEBRTC_WIN)
#pragma comment(lib, "onnxruntime.lib")
#endif

namespace pns 
{
VoiceFilter::VoiceFilter(const ORTCHAR_T* modelPath, size_t featureDim, size_t embedderDim, size_t hiddenNumber) :  startCount_(0), featureDim_(featureDim) {
    std::array<int64_t, 3> specturmShape{1, 9, static_cast<long long>(featureDim)};
    std::array<int64_t, 3> embedderShape{1, 1, static_cast<long long>(embedderDim)};
    std::array<int64_t, 3> hiddenShape{1, 1, static_cast<long long>(hiddenNumber)};
    std::array<int64_t, 3> cellShape{1, 1, static_cast<long long>(hiddenNumber)};
    std::array<int64_t, 3> outputShape{1, 1, static_cast<long long>(featureDim)};
    specturmData_.resize(9 * featureDim, 0.0f);
    embedderData_.resize(embedderDim, 0.0f);
    hiddenData_.resize(hiddenNumber, 0.0f);
    cellData_.resize(hiddenNumber, 0.0f);
    outputData_.resize(featureDim, 0.0f);

    session_ = Ort::Session(env_, modelPath, Ort::SessionOptions{nullptr});

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, specturmData_.data(), specturmData_.size(), specturmShape.data(), specturmShape.size()));
    inputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, embedderData_.data(), embedderData_.size(), embedderShape.data(), embedderShape.size()));
    inputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, hiddenData_.data(), hiddenData_.size(), hiddenShape.data(), hiddenShape.size()));
    inputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, cellData_.data(), cellData_.size(), cellShape.data(), cellShape.size()));

    outputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputData_.data(), outputData_.size(), outputShape.data(), outputShape.size()));
    outputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, hiddenData_.data(), hiddenData_.size(), hiddenShape.data(), hiddenShape.size()));
    outputTensors_.push_back(Ort::Value::CreateTensor<float>(memoryInfo, cellData_.data(), cellData_.size(), cellShape.data(), cellShape.size()));
}

VoiceFilter::~VoiceFilter() {}

void VoiceFilter::setDVector(std::vector<float>& dVector) {
    std::copy(dVector.begin(), dVector.end(), embedderData_.begin());
}

std::vector<float> VoiceFilter::process(std::shared_ptr<std::vector<float> > specturm) {
    const char* inputNames[] = {"onnx::Unsqueeze_0", "onnx::Slice_1", "onnx::LSTM_2", "onnx::LSTM_3"};
    const char* outputNames[] = {"367", "354", "355"};

    startCount_ += 1;

    std::copy(specturmData_.begin() + featureDim_, specturmData_.end(), specturmData_.begin());
    std::copy(specturm->begin(), specturm->end(), specturmData_.begin() + featureDim_ * 8);
    if (startCount_ < 9) {
        return outputData_;
    }
    session_.Run(Ort::RunOptions{nullptr}, inputNames, inputTensors_.data(), 4, outputNames, outputTensors_.data(), 3);   

    return outputData_;
}
} // namespace pns

//#define UNITTEST
#ifdef UNITTEST

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    std::unique_ptr<pns::VoiceFilter> voiceFilter_;
    voiceFilter_.reset(new pns::VoiceFilter(513, 256, 600));
    std::cout << "voice filter instance create OK" << std::endl;

    // mask file
    std::ofstream outputStream("output.bin", std::ios_base::out | std::ios_base::binary);

    // load npy datas
    uint8_t buf[10];
    std::ifstream npyStream("fullFeature_0.npy", std::ios_base::in | std::ios_base::binary);
    npyStream.read((char *)&buf[0], 10);                                                                                                                                                                                                                                                                                          
    std::cout << (uint16_t)buf[8] << std::endl;
    npyStream.seekg((int)buf[8] + 10);

    std::vector<float> dataBuf(384600, 0.0f);

    std::vector<float> input(513, 0.0f);
    std::vector<float> dvector(256, 0.0f);
    for (int i = 0; i < 16; i++) {
        npyStream.read((char*)dataBuf.data(), 384600 * sizeof(float));
        for (int j = 0; j < 300; j++) {
            size_t startIter1 = j * 1282;
            size_t endIter1 = j * 1282 + 513;
            std::copy(dataBuf.begin() + startIter1, dataBuf.begin() + endIter1, input.begin());
            size_t startIter2 = j * 1282 + 513;
            size_t endIter2 = j * 1282 + 769;
            std::copy(dataBuf.begin() + startIter2, dataBuf.begin() + endIter2, dvector.begin());
            voiceFilter_->setDVector(dvector);

            std::chrono::steady_clock::time_point beginTime = std::chrono::steady_clock::now();
            std::vector<float> output = voiceFilter_->process(input);
            std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
            std::chrono::steady_clock::duration time_span = endTime - beginTime;
            double nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
            std::cout << nseconds << std::endl;
            outputStream.write((const char*)output.data(), 513 * sizeof(float));
        }
    }
    
    input.clear();
    dvector.clear();
    dataBuf.clear();

    return 0;
}

#endif // UNITTEST
