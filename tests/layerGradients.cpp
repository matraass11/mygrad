#include "gradientCheck.hpp"

using namespace mygrad;
using namespace gradcheck;

// Configurations follow ticket 003: roughly two per layer, chosen to move the
// index math around rather than to make the tensors big. Batch is always > 1.

TEST_CASE("LinearLayer gradients", "[gradcheck]") {
    setSeed(1);

    SECTION("small batch") {
        LinearLayer layer(4, 5);
        Tensor input(spreadValues(3 * 4, 11), {3, 4});

        checkLayerGradients(layer, input, 101);
    }

    // Work is split over batch rows, and every row accumulates into every weight
    // row and bias. Three rows barely give the threads a chance to overlap, so
    // this wider batch is what actually puts the guarded accumulation under load
    // in the N-thread run — a lost update shows up as a plainly wrong gradient.
    SECTION("batch wide enough to spread across the pool") {
        LinearLayer layer(6, 6);
        Tensor input(spreadValues(64 * 6, 23), {64, 6});

        checkLayerGradients(layer, input, 113);
    }
}


TEST_CASE("Conv2d gradients", "[gradcheck]") {
    setSeed(2);

    SECTION("stride 1, no padding") {
        Conv2d layer(2, 3, 3, 1, 0);
        Tensor input(spreadValues(3 * 2 * 5 * 5, 12), {3, 2, 5, 5});

        checkLayerGradients(layer, input, 102);
    }

    SECTION("stride 2, padding 1") {
        Conv2d layer(2, 3, 3, 2, 1);
        Tensor input(spreadValues(3 * 2 * 5 * 5, 13), {3, 2, 5, 5});

        checkLayerGradients(layer, input, 103);
    }
}


TEST_CASE("ReLU gradients", "[gradcheck]") {
    ReLU layer;
    Tensor input(valuesAwayFromZero(3 * 4, 14), {3, 4}); // the kink at zero is avoided, not asserted

    checkLayerGradients(layer, input, 104);
}


TEST_CASE("Sigmoid gradients", "[gradcheck]") {
    Sigmoid layer;
    Tensor input(spreadValues(3 * 4, 15), {3, 4});

    checkLayerGradients(layer, input, 105);
}


TEST_CASE("MaxPool2d gradients", "[gradcheck]") {

    SECTION("dimensions divisible by the kernel") {
        MaxPool2d layer(2);
        Tensor input(wellSeparatedValues(2 * 2 * 4 * 4, 16), {2, 2, 4, 4});

        checkLayerGradients(layer, input, 106);
    }

    SECTION("dimensions not divisible by the kernel") { // the trailing partial window is dropped
        MaxPool2d layer(2);
        Tensor input(wellSeparatedValues(2 * 2 * 5 * 5, 17), {2, 2, 5, 5});

        checkLayerGradients(layer, input, 107);
    }
}


TEST_CASE("Upsample gradients", "[gradcheck]") {

    SECTION("scaling factor 2") {
        Upsample layer(2);
        Tensor input(spreadValues(2 * 2 * 3 * 3, 18), {2, 2, 3, 3});

        checkLayerGradients(layer, input, 108);
    }

    SECTION("scaling factor 3") {
        Upsample layer(3);
        Tensor input(spreadValues(2 * 2 * 3 * 3, 19), {2, 2, 3, 3});

        checkLayerGradients(layer, input, 109);
    }
}


TEST_CASE("Reshape gradients", "[gradcheck]") {

    SECTION("fixed dimensions") {
        Reshape layer(TensorDims{6, 4}); // spelled out: {6, 4} alone reads as (dimensions, freeDimension)
        Tensor input(spreadValues(3 * 8, 20), {3, 8});

        checkLayerGradients(layer, input, 110);
    }

    SECTION("free dimension") { // the form the cats example uses
        Reshape layer({1, 4}, 0);
        Tensor input(spreadValues(3 * 8, 21), {3, 8});

        checkLayerGradients(layer, input, 111);
    }
}


TEST_CASE("Reparameterize gradients", "[gradcheck]") {
    Reparameterize layer;
    // columns are interleaved (mean, logvariance) pairs, whatever the layer's own
    // error message says about n means followed by n logvariances
    Tensor input(spreadValues(3 * 4, 22), {3, 4});

    // Without resetting the stream before every single forward, the two sides of
    // the difference draw different epsilons and the check measures noise.
    checkLayerGradients(layer, input, 112, [&layer] { layer.setSeed(7); });
}
