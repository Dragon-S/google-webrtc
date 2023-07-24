#include <iostream>
#include <memory>
#include <vector>

#include "nn.h"
#include "fft.h"

namespace nnPlc
{
class Plc {
public:
    Plc() = default;
    ~Plc();

    int initialize();

    void process(const float* input, float* output);

private:
    void phaseReconstruct();

private:
    const size_t fftSize_ = 256;
    const size_t processSize_ = 128;
    const size_t blockSize_ = 160;
    const size_t overlapSize_ = 96;

    size_t plcCount_ = 0;

    std::vector<float> phaseBase_;
    std::vector<float> phaseInc_;
    std::vector<float> real_;
    std::vector<float> imag_;
    std::vector<float> logX_;
    std::vector<float> ampY_;
    std::vector<float> phaY_;
    
    std::unique_ptr<Fft> fft_ = nullptr;
    std::unique_ptr<NN> nn_ = nullptr;    
};    
} // namespace nnPlc
