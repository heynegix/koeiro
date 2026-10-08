#include <signalsmith-stretch/signalsmith-stretch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <vector>
#include <xmmintrin.h>
#include "tone.h"
#include "limiter.h"

// C ABI version 1. All arrays are allocated at create time. The core is owned
// by the audio callback after prepare; parameters are copied in by that thread.
struct Parameters {
    float pitch, formant, brightness, wet;
    int32_t enabled, lowCut, limiter;
};

struct FloatMode {
    unsigned previous = _mm_getcsr();
    FloatMode() { _mm_setcsr(previous | 0x8040); }
    ~FloatMode() { _mm_setcsr(previous); }
};

struct NativeTiming {
    uint64_t count, totalNs, maximumNs, lastNs;
};
struct TimingCounters {
    std::atomic<uint64_t> count{0}, total{0}, maximum{0}, last{0};
};
struct TimingGuard {
    TimingCounters &stats;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    explicit TimingGuard(TimingCounters &s) : stats(s) {}
    ~TimingGuard() {
        auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count());
        stats.total.fetch_add(ns, std::memory_order_relaxed);
        stats.last.store(ns, std::memory_order_relaxed);
        stats.maximum.store(std::max(ns, stats.maximum.load(std::memory_order_relaxed)), std::memory_order_relaxed);
        stats.count.fetch_add(1, std::memory_order_relaxed);
    }
};

struct DSP {
    signalsmith::stretch::SignalsmithStretch<float> stretch{1};
    Tone tone;
    Limiter limiter;
    TimingCounters timing;
    int rate, maxFrames, latency, dryPosition = 0, sinceEnable = 0;
    std::vector<float> input, processed, delay;
    float pitch = 3, formant = 2, brightness = 60, wet = 1, lowCut = 1, limited = 1, effect = 0;
    bool wasEnabled = false;
    explicit DSP(int sampleRate, int frames, int quality)
        : tone(sampleRate), limiter(static_cast<float>(sampleRate)), rate(sampleRate), maxFrames(frames),
          input(frames), processed(frames) {
        int window = quality == 0 ? 2048 : 4096;
        int hop = quality == 0 ? 128 : 256;
        stretch.configure(1, window, hop, false);
        // A conservative male-speech register gives the spectral-envelope
        // smoother a stable scale. Automatic peak-only F0 guesses can mistake
        // a strong vowel resonance for the fundamental with very short windows.
        stretch.setFormantBase(150.0f/sampleRate);
        latency = stretch.inputLatency() + stretch.outputLatency();
        delay.resize(std::max(1, latency), 0);
        // Exercise FFT/formant code paths before capture. Then reset all state.
        stretch.setTransposeSemitones(3);
        stretch.setFormantSemitones(2, true);
        for (int i = 0; i < frames; ++i) input[i] = 0.05f*std::sin(i*0.43f);
        const float *in[] = {input.data()}; float *out[] = {processed.data()};
        for (int i = 0; i < window*4; i += frames) stretch.process(in, frames, out, frames);
        reset();
    }
    void reset() {
        stretch.reset(); tone.reset(); limiter.reset();
        std::fill(delay.begin(), delay.end(), 0.0f);
        dryPosition = sinceEnable = 0; effect = 0; wasEnabled = false;
    }
    bool valid(const Parameters &p) {
        return std::isfinite(p.pitch) && std::abs(p.pitch) <= 12
            && std::isfinite(p.formant) && std::abs(p.formant) <= 6
            && std::isfinite(p.brightness) && p.brightness >= 0 && p.brightness <= 100
            && std::isfinite(p.wet) && p.wet >= 0 && p.wet <= 1;
    }
    void process(const float *source, float *output, int frames, const Parameters &p) {
        TimingGuard duration(timing);
        // Save/restore the host thread FP mode; avoid denormal CPU spikes in EQ.
        FloatMode fpMode;
        if (!p.enabled && effect < 0.00001f) {
            effect = 0; wasEnabled = false;
            for (int i = 0; i < frames; ++i) output[i] = std::isfinite(source[i]) ? source[i] : 0;
            return;  // Exact, zero-delay Original path.
        }
        if (p.enabled && !wasEnabled && effect == 0) {
            reset();
            pitch = p.pitch; formant = p.formant; brightness = p.brightness;
            wet = p.wet; lowCut = static_cast<float>(p.lowCut != 0); limited = static_cast<float>(p.limiter != 0);
        }
        wasEnabled = p.enabled != 0;
        for (int i = 0; i < frames; ++i) input[i] = std::isfinite(source[i]) ? std::clamp(source[i], -8.0f, 8.0f) : 0;
        // 64-sample control quantum, independent of the host callback Buffer.
        for (int offset = 0; offset < frames; offset += 64) {
            int count = std::min(64, frames-offset);
            float control = 1-std::exp(-count/(0.030f*rate));
            pitch += control*(p.pitch-pitch); formant += control*(p.formant-formant);
            brightness += control*(p.brightness-brightness); wet += control*(p.wet-wet);
            lowCut += control*(static_cast<float>(p.lowCut != 0)-lowCut);
            limited += control*(static_cast<float>(p.limiter != 0)-limited);
            stretch.setTransposeSemitones(pitch);
            // Compensation removes the pitch-induced envelope shift. Formant
            // controls the envelope independently while harmonic spacing shifts.
            stretch.setFormantSemitones(formant, true);
            tone.configure(brightness);
            const float *in[] = {input.data()+offset}; float *out[] = {processed.data()+offset};
            stretch.process(in, count, out, count);  // Equal lengths: rate stays 1.
            float fadeStep = 1/(0.012f*rate);
            for (int i = offset; i < offset+count; ++i) {
                float dry = delay[dryPosition]; delay[dryPosition] = input[i];
                dryPosition = (dryPosition+1)%latency;
                float value = tone.process(processed[i], lowCut);
                float effected = dry*(1-wet) + value*wet;
                float protectedValue = limiter.process(effected);
                effected += limited*(protectedValue-effected);
                sinceEnable = std::min(latency+1, sinceEnable+1);
                bool ready = sinceEnable > latency;
                float target = p.enabled && ready ? 1.0f : 0.0f;
                effect += std::clamp(target-effect, -fadeStep, fadeStep);
                float mixed = input[i] + effect*(effected-input[i]);
                output[i] = std::isfinite(mixed) ? std::clamp(mixed, -1.0f, 1.0f) : 0;
            }
        }
    }
};

#define API extern "C" __declspec(dllexport)
API int avc_abi_version() { return 1; }
API void *avc_create(int rate, int maxFrames, int quality) noexcept {
    if ((rate != 44100 && rate != 48000) || maxFrames < 1 || maxFrames > 1024 || quality < 0 || quality > 1) return nullptr;
    try { return new DSP(rate, maxFrames, quality); } catch (...) { return nullptr; }
}
API void avc_destroy(void *handle) noexcept { delete static_cast<DSP *>(handle); }
API int avc_latency(void *handle) noexcept { return handle ? static_cast<DSP *>(handle)->latency : 0; }
API int avc_timing(void *handle, NativeTiming *result) noexcept {
    if (!handle || !result) return 1;
    auto &s = static_cast<DSP *>(handle)->timing;
    result->count = s.count.load(std::memory_order_relaxed);
    result->totalNs = s.total.load(std::memory_order_relaxed);
    result->maximumNs = s.maximum.load(std::memory_order_relaxed);
    result->lastNs = s.last.load(std::memory_order_relaxed);
    return 0;
}
API void avc_reset(void *handle) noexcept { if (handle) static_cast<DSP *>(handle)->reset(); }
API int avc_process(void *handle, const float *input, float *output, int frames, const Parameters *parameters) noexcept {
    if (!handle || !input || !output || !parameters || frames < 1) return 1;
    auto &dsp = *static_cast<DSP *>(handle);
    if (frames > dsp.maxFrames || !dsp.valid(*parameters)) return 1;
    try { dsp.process(input, output, frames, *parameters); return 0; }
    catch (...) { std::fill(output, output+frames, 0.0f); return 2; }
}
