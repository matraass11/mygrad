#pragma once

// Finite-difference gradient checking, per the methodology fixed in ticket 003.
//
// The scalar being differentiated is f = dot(v, output) for a fixed random v.
// Not a plain sum: sum is permutation invariant, so it cannot see a layer that
// scrambles the order of its own output — which is exactly how the index-heavy
// layers (im2col, Upsample, Reshape, MaxPool2d) go wrong.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "mygrad/mygrad.hpp"

namespace gradcheck {

using mygrad::Layer;
using mygrad::Model;
using mygrad::Tensor;
using mygrad::TensorDims;

// dtype is double, so eps is ~2.2e-16 and a central difference is at its best
// around 6e-6 with an error floor near 1e-10 — orders below the tolerance.
constexpr dtype step = 1e-6;

// Mixed absolute and relative: exact-zero gradients are routine here (ReLU below
// zero, every non-max element of a pool window), and a purely relative check
// divides zero by zero on them. Real errors are factor sized, never 1e-7 sized.
constexpr dtype absoluteTolerance = 1e-7;
constexpr dtype relativeTolerance = 1e-6;


inline std::vector<dtype> spreadValues( size_t count, unsigned seed, dtype magnitude = 2 ) {
    std::mt19937 generator(seed); // test-local, never the library's static generator
    std::uniform_real_distribution<dtype> spread(-magnitude, magnitude);

    std::vector<dtype> values(count);
    for (dtype& value : values) { value = spread(generator); }
    return values;
}


// A central difference is wrong at a kink whichever subgradient the code picks:
// at x = 0 ReLU's analytic gradient is 1 while the difference quotient gives 0.5.
// So the kinks are kept out of the generated data, 1000 steps clear of zero.
inline std::vector<dtype> valuesAwayFromZero( size_t count, unsigned seed, dtype margin = 1e-3 ) {
    std::vector<dtype> values = spreadValues(count, seed);
    for (dtype& value : values) {
        if (std::abs(value) < margin) { value = std::copysign(2 * margin, value); }
    }
    return values;
}


// Every pool window needs an unambiguous winner. A near-tie breaks the numeric
// derivative just as an exact tie does: if the gap is under the step, perturbing
// the maximum flips the argmax and the two forwards measure different functions.
// A shuffled arithmetic sequence puts every pair of values a whole spacing apart.
inline std::vector<dtype> wellSeparatedValues( size_t count, unsigned seed, dtype spacing = 0.25 ) {
    std::vector<dtype> values(count);
    for (size_t i = 0; i < count; i++) {
        values[i] = static_cast<dtype>(i) * spacing - static_cast<dtype>(count) * spacing / 2;
    }
    std::shuffle(values.begin(), values.end(), std::mt19937(seed));
    return values;
}


// Inside a composite the kink-sensitive values are computed, not generated, so
// the margins have to be asserted rather than arranged. Worth asserting loudly:
// biases initialize to zero, so a weight seed that clips a whole ReLU row lands
// the next layer's pre-activation on exactly zero and quietly poisons the
// difference quotient there.
inline void requireAwayFromZero( const Tensor& tensor, const std::string& what, dtype margin = 1e-3 ) {
    dtype closest = std::numeric_limits<dtype>::max();
    for (size_t i = 0; i < tensor.length; i++) { closest = std::min(closest, std::abs(tensor.data[i])); }

    INFO(what << " comes within " << closest << " of the kink at zero");
    REQUIRE(closest > margin);
}


// Same idea for pooling: a window whose top two values are closer together than
// the step changes its argmax under perturbation, and the two forwards then
// measure different functions.
inline void requireWindowsHaveClearWinners( const Tensor& tensor, size_t kernelSize, dtype margin = 1e-3 ) {
    dtype narrowestGap = std::numeric_limits<dtype>::max();

    for (size_t picture = 0; picture < tensor.dimensions[0]; picture++) {
        for (size_t channel = 0; channel < tensor.dimensions[1]; channel++) {
            for (size_t row = 0; row + kernelSize <= tensor.dimensions[2]; row += kernelSize) {
                for (size_t column = 0; column + kernelSize <= tensor.dimensions[3]; column += kernelSize) {

                    dtype best = -std::numeric_limits<dtype>::max();
                    dtype runnerUp = -std::numeric_limits<dtype>::max();

                    for (size_t withinRow = row; withinRow < row + kernelSize; withinRow++) {
                        for (size_t withinColumn = column; withinColumn < column + kernelSize; withinColumn++) {
                            const dtype value = tensor.at({picture, channel, withinRow, withinColumn});
                            if (value > best) { runnerUp = best, best = value; }
                            else if (value > runnerUp) { runnerUp = value; }
                        }
                    }

                    // A window a ReLU has clipped flat is not a real tie: whichever
                    // element the pool picks, the ReLU behind it blocks the gradient,
                    // so both sides of the difference agree. That holds only while
                    // those inputs stay clear of the kink, which is asserted separately.
                    if (best == 0 and runnerUp == 0) { continue; }

                    narrowestGap = std::min(narrowestGap, best - runnerUp);
                }
            }
        }
    }

    INFO("the tightest pool window separates its top two values by " << narrowestGap);
    REQUIRE(narrowestGap > margin);
}


struct Subject {
    std::function<dtype()> value;   // one forward pass, returning the scalar being differentiated
    std::function<void()> analytic; // zeroGrad, forward, seed the output grads, backward
    std::vector<Tensor*> checked;   // perturbed element by element, their grads compared
    std::vector<std::string> names; // one per checked tensor, for the failure message
};


inline void requireGradientsMatch( const Subject& subject ) {
    subject.analytic();

    std::vector<std::vector<dtype>> analyticGrads;
    for (const Tensor* tensor : subject.checked) {
        analyticGrads.emplace_back(tensor->grads.get(), tensor->grads.get() + tensor->length);
    }

    dtype worstExcess = 0; // how far past tolerance the worst element sits
    std::string worstOffender = "every element within tolerance";

    for (size_t tensorIndex = 0; tensorIndex < subject.checked.size(); tensorIndex++) {
        Tensor& tensor = *subject.checked[tensorIndex];

        for (size_t element = 0; element < tensor.length; element++) {
            const dtype original = tensor.data[element];

            tensor.data[element] = original + step;
            const dtype valueAbove = subject.value();
            tensor.data[element] = original - step;
            const dtype valueBelow = subject.value();
            tensor.data[element] = original;

            const dtype numeric = (valueAbove - valueBelow) / (2 * step);
            const dtype analytic = analyticGrads[tensorIndex][element];
            const dtype excess = std::abs(analytic - numeric)
                                 - (absoluteTolerance + relativeTolerance * std::abs(numeric));

            if (excess > worstExcess) {
                worstExcess = excess;

                std::ostringstream offender;
                offender << std::setprecision(12) << subject.names[tensorIndex]
                         << ", element " << element
                         << ": analytic " << analytic << " vs numeric " << numeric;
                worstOffender = offender.str();
            }
        }
    }

    INFO("worst gradient mismatch: " << worstOffender);
    REQUIRE(worstExcess <= 0);
}


// Checks a layer's input gradients and its parameter gradients in one pass —
// Conv2d computes both inside one backward, so checking either alone leaves the
// other unverified. beforeForward exists for Reparameterize, whose epsilons have
// to be redrawn identically for both sides of the difference.
inline void checkLayerGradients( Layer& layer, Tensor& input, unsigned seed,
                                 const std::function<void()>& beforeForward = [] {} ) {
    beforeForward();
    layer.forward(input); // shapes are inferred lazily, so the output has no length before this

    const std::vector<dtype> outputSeed = spreadValues(layer.outputTensor.length, seed, 1);

    Subject subject;

    subject.value = [&layer, &input, outputSeed, &beforeForward] {
        beforeForward();
        layer.forward(input);
        return std::inner_product(outputSeed.begin(), outputSeed.end(),
                                  layer.outputTensor.data.get(), static_cast<dtype>(0));
    };

    subject.analytic = [&layer, &input, outputSeed, &beforeForward] {
        layer.zeroGrad(); // grads accumulate, so a stale pass would compound into this one
        input.zeroGrad();
        beforeForward();
        layer.forward(input);
        std::copy(outputSeed.begin(), outputSeed.end(), layer.outputTensor.grads.get());
        layer.backward();
    };

    subject.checked.push_back(&input);
    subject.names.push_back("input");

    const std::vector<Tensor*> parameters = layer.parameterTensors();
    for (size_t i = 0; i < parameters.size(); i++) {
        subject.checked.push_back(parameters[i]);
        subject.names.push_back("parameter tensor " + std::to_string(i));
    }

    requireGradientsMatch(subject);
}


// Losses already produce a scalar, so there is no projection to seed — their
// output gradient is simply 1.
inline void checkLossGradients( const std::function<dtype()>& lossValue,
                                const std::function<void()>& lossBackward,
                                Tensor& lossInput ) {
    Subject subject;

    subject.value = lossValue;
    subject.analytic = [&lossValue, &lossBackward, &lossInput] {
        lossInput.zeroGrad();
        lossValue(); // backward is only valid straight after its own operator()
        lossBackward();
    };
    subject.checked = { &lossInput };
    subject.names = { "loss input" };

    requireGradientsMatch(subject);
}


// Driven through Model rather than by hand-chaining layers: the wiring itself is
// what these checks are for — the raw input-tensor handoff between layers, the
// reverse iteration in Model::backward, and whether zeroGrad reaches every
// gradient-carrying tensor. A per-layer suite passes with all of that broken.
inline void checkModelGradients( Model& model, Tensor& input,
                                 const std::function<dtype()>& lossValue,
                                 const std::function<void()>& lossBackward ) {
    Subject subject;

    subject.value = [&model, &input, &lossValue] {
        model.forward(input);
        return lossValue();
    };

    subject.analytic = [&model, &input, &lossValue, &lossBackward] {
        model.zeroGrad();
        input.zeroGrad();
        model.forward(input);
        lossValue();
        lossBackward();
        model.backward();
    };

    subject.checked.push_back(&input);
    subject.names.push_back("input");

    for (size_t i = 0; i < model.parameters.size(); i++) {
        subject.checked.push_back(model.parameters[i]);
        subject.names.push_back("model parameter tensor " + std::to_string(i));
    }

    requireGradientsMatch(subject);
}

} // namespace gradcheck
