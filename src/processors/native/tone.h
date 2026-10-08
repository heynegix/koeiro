#pragma once
#include <algorithm>
#include <cmath>

struct Biquad {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;
    void coefficients(double B0, double B1, double B2, double A0, double A1, double A2) {
        b0 = B0/A0; b1 = B1/A0; b2 = B2/A0; a1 = A1/A0; a2 = A2/A0;
    }
    float process(float x) {
        double y = b0*x + z1;
        z1 = b1*x - a1*y + z2;
        z2 = b2*x - a2*y;
        return static_cast<float>(y);
    }
    void reset() { z1 = z2 = 0; }
};

struct Tone {
    Biquad highpass, low, presence, high;
    double rate;
    float lastBrightness = -1;
    explicit Tone(double sampleRate) : rate(sampleRate) {
        double w = 2*3.141592653589793*90/rate, c = std::cos(w), s = std::sin(w);
        double alpha = s/(2*std::sqrt(0.5));
        highpass.coefficients((1+c)/2, -(1+c), (1+c)/2, 1+alpha, -2*c, 1-alpha);
        configure(50);
    }
    void shelf(Biquad &filter, double frequency, double db, bool upper) {
        double A = std::pow(10, db/40), w = 2*3.141592653589793*frequency/rate;
        double c = std::cos(w), alpha = std::sin(w)/std::sqrt(2.0), beta = 2*std::sqrt(A)*alpha;
        if (upper) {
            filter.coefficients(A*((A+1)+(A-1)*c+beta), -2*A*((A-1)+(A+1)*c),
                A*((A+1)+(A-1)*c-beta), (A+1)-(A-1)*c+beta,
                2*((A-1)-(A+1)*c), (A+1)-(A-1)*c-beta);
        } else {
            filter.coefficients(A*((A+1)-(A-1)*c+beta), 2*A*((A-1)-(A+1)*c),
                A*((A+1)-(A-1)*c-beta), (A+1)+(A-1)*c+beta,
                -2*((A-1)+(A+1)*c), (A+1)+(A-1)*c-beta);
        }
    }
    void configure(float brightness) {
        if (std::abs(brightness - lastBrightness) < 0.001f) return;
        lastBrightness = brightness;
        double amount = (brightness - 50)/50;
        shelf(low, 300, -2*amount, false);
        shelf(high, 3800, 4*amount, true);
        double A = std::pow(10, 2*amount/40), w = 2*3.141592653589793*2500/rate;
        double alpha = std::sin(w)/(2*0.8), c = std::cos(w);
        presence.coefficients(1+alpha*A, -2*c, 1-alpha*A, 1+alpha/A, -2*c, 1-alpha/A);
    }
    float process(float x, float lowCutMix) {
        float hp = highpass.process(x);
        return high.process(presence.process(low.process(x + lowCutMix*(hp-x))));
    }
    void reset() { highpass.reset(); low.reset(); presence.reset(); high.reset(); }
};
