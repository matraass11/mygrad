#include <catch2/catch_test_macros.hpp>

#include "mygrad/mygrad.hpp"

using namespace mygrad;

// Finite differences can never settle what happens at a kink — the difference
// quotient there matches no subgradient at all. These two pin the choices the
// code makes instead, so that a later edit changing one of them is deliberate.

TEST_CASE("ReLU passes the gradient through at exactly zero", "[conventions]") {
    ReLU layer;
    Tensor input({-1.0, 0.0, 1.0}, {1, 3});

    layer.forward(input);
    layer.outputTensor.grads[0] = 5;
    layer.outputTensor.grads[1] = 5;
    layer.outputTensor.grads[2] = 5;
    layer.backward();

    REQUIRE(input.grads[0] == 0); // below zero, blocked
    REQUIRE(input.grads[1] == 5); // at zero, passed through: forward and backward both test >= 0
    REQUIRE(input.grads[2] == 5);
}


TEST_CASE("MaxPool2d routes an exact tie to the first element", "[conventions]") {
    MaxPool2d layer(2);
    Tensor input({1.0, 1.0, 1.0, 1.0}, {1, 1, 2, 2}); // one window, four identical maxima

    layer.forward(input);
    layer.outputTensor.grads[0] = 3;
    layer.backward();

    REQUIRE(input.grads[0] == 3); // strict > in both the forward and the backward scan
    REQUIRE(input.grads[1] == 0);
    REQUIRE(input.grads[2] == 0);
    REQUIRE(input.grads[3] == 0);
}
