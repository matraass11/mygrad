#include "gradientCheck.hpp"

using namespace mygrad;
using namespace gradcheck;

TEST_CASE("CrossEntropyLoss gradients", "[gradcheck]") {
    CrossEntropyLoss loss;
    Tensor logits(spreadValues(4 * 3, 30), {4, 3});
    const Tensor labels({0, 2, 1, 2}, {4});

    // No margin is needed around the max subtraction inside the loss: it is
    // log-sum-exp stabilization and cancels out of the value, so an argmax flip
    // under perturbation changes nothing.
    checkLossGradients([&loss, &logits, &labels] { return loss(logits, labels); },
                       [&loss] { loss.backward(); },
                       logits);
}


TEST_CASE("MSEloss gradients", "[gradcheck]") {
    Tensor outputs(spreadValues(3 * 4, 31), {3, 4});
    const Tensor labels(spreadValues(3 * 4, 32), {3, 4});

    SECTION("reduction sum") { // divides by the batch
        MSEloss loss("sum");

        checkLossGradients([&loss, &outputs, &labels] { return loss(outputs, labels); },
                           [&loss] { loss.backward(); },
                           outputs);
    }

    SECTION("reduction mean") { // divides by every element
        MSEloss loss("mean");

        checkLossGradients([&loss, &outputs, &labels] { return loss(outputs, labels); },
                           [&loss] { loss.backward(); },
                           outputs);
    }
}


TEST_CASE("KLdivWithStandardNormal gradients", "[gradcheck]") {
    KLdivWithStandardNormal loss;
    Tensor distribution(spreadValues(3 * 4, 33), {3, 4}); // interleaved (mean, logvariance)

    SECTION("beta 1") {
        checkLossGradients([&loss, &distribution] { return loss(distribution, 1); },
                           [&loss] { loss.backward(); },
                           distribution);
    }

    SECTION("beta other than 1") { // backward folds beta into its divisor
        checkLossGradients([&loss, &distribution] { return loss(distribution, 0.7); },
                           [&loss] { loss.backward(); },
                           distribution);
    }

    SECTION("beta 0") { // weighting the divergence out entirely, which used to throw
        checkLossGradients([&loss, &distribution] { return loss(distribution, 0); },
                           [&loss] { loss.backward(); },
                           distribution);
    }
}
