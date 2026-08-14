#include "gradientCheck.hpp"

using namespace mygrad;
using namespace gradcheck;

// Two composites, so that every layer appears inside a chain at least once and
// Model's own wiring is under test rather than each backward in isolation.

TEST_CASE("MLP composite gradients", "[gradcheck]") {
    setSeed(2); // seed 3 clips a whole hidden row, landing the next pre-activation on exactly zero

    Model model(
        LinearLayer(4, 6),
        ReLU(),
        LinearLayer(6, 6),
        ReLU(),
        LinearLayer(6, 3)
    );

    Tensor input(spreadValues(4 * 4, 40), {4, 4});
    const Tensor labels({0, 2, 1, 1}, {4});

    CrossEntropyLoss loss;
    Tensor& outputs = model.forward(input); // the reference survives an output reallocation

    requireAwayFromZero(model.layers[0].outputTensor, "the first ReLU's input");
    requireAwayFromZero(model.layers[2].outputTensor, "the second ReLU's input");

    checkModelGradients(model, input,
                        [&loss, &outputs, &labels] { return loss(outputs, labels); },
                        [&loss] { loss.backward(); });
}


TEST_CASE("convolutional composite gradients", "[gradcheck]") {
    setSeed(4);

    Model model(
        Conv2d(2, 3, 3, 1, 0),
        ReLU(),
        MaxPool2d(2),
        Reshape({1, 12}, 0),
        LinearLayer(12, 2)
    );

    Tensor input(spreadValues(2 * 2 * 6 * 6, 41), {2, 2, 6, 6});
    const Tensor labels(spreadValues(2 * 2, 42), {2, 2});

    MSEloss loss("mean");
    Tensor& outputs = model.forward(input);

    requireAwayFromZero(model.layers[0].outputTensor, "the ReLU's input");
    requireWindowsHaveClearWinners(model.layers[1].outputTensor, 2);

    checkModelGradients(model, input,
                        [&loss, &outputs, &labels] { return loss(outputs, labels); },
                        [&loss] { loss.backward(); });
}
