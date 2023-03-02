#ifndef PERSONAL_NS_H_
#define PERSONAL_NS_H_

#include <memory>
#include <vector>

namespace pns
{

class PnsFft;
class MelFeature;
class Embedder;
class VoiceFilter;

enum PersonalNsMode {
    normal = 0,
    slight,
};

class PersonalNs {
public:
    PersonalNs() = default;

    virtual ~PersonalNs() = default;

    static PersonalNs* createPns();

    virtual int processFrame(float* noisyFrame) = 0;

    virtual int extractEmbedder() = 0;
};

class PersonalNsImpl: public PersonalNs {
public:
    PersonalNsImpl();

    ~PersonalNsImpl();

    virtual int processFrame(float* frame) override;

    virtual int extractEmbedder() override;

private:
    int saveEmbedder(std::vector<float> dVector);

    int loadEmbedder();

private:
    const size_t frameSize_ = 160;

    const size_t windowSize_ = 400;

    const size_t reciveSize_ = 480;

    const size_t fftSize_ = 1024;

    const size_t specturmSize_ = 513;

    const size_t hiddenNumber_ = 600;

    const size_t melFrameSize_ = 160;

    const size_t melFftSize_ = 512;

    const float melLowsetEnergy_= 2.0f;

    const size_t melNumber_ = 40;

    const size_t embedderDim_ = 256;

private:
    size_t startPhase_;

    size_t writePosition_;

    size_t readPosition_;

    std::vector<float> dVector_;

    std::vector<std::shared_ptr<std::vector<float> > > realLatency_;

    std::vector<std::shared_ptr<std::vector<float> > > imagLatency_;

    std::vector<std::shared_ptr<std::vector<float> > > ampLatency_;

    std::vector<float> recvBuffer_;

    std::vector<float> synBuffer_;

    std::vector<float> synWindow_;

    std::unique_ptr<PnsFft> pnsFft_;

    std::unique_ptr<MelFeature> melFeature_;

    std::unique_ptr<Embedder> embedder_;

    std::unique_ptr<VoiceFilter> voiceFilter_;

    PersonalNsMode mode_;

    bool voicefilterTurnOn_;

    float smoothedRms_;

    std::vector<float> noisyRms_;

    size_t slightHurtCount_;

    size_t seriousHurtCount_;
};    
} // namespace pns


#endif // PERSONAL_NS_H_