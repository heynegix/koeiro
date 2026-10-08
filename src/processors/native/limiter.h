#pragma once
#include <algorithm>
#include <cmath>

struct Limiter {
    float gain = 1;
    float release;
    explicit Limiter(float rate) : release(1 - std::exp(-1/(0.080f*rate))) {}
    float process(float x) {
        float peak = std::abs(x);
        float target = peak > 0.95f ? 0.95f/peak : 1.0f;
        gain = target < gain ? target : gain + release*(target-gain);
        return x*gain;
    }
    void reset() { gain = 1; }
};
