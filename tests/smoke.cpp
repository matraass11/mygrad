// Placeholder suite: it exists so the test target, ctest wiring and both thread
// configurations are exercised before the gradient checks land (ticket 007).

#include <catch2/catch_test_macros.hpp>

#include "mygrad/mygrad.hpp"

using namespace mygrad;

TEST_CASE("a model forwards to the shape its last layer declares", "[smoke]") {
    Model model(
        LinearLayer(4, 3),
        ReLU(),
        LinearLayer(3, 2)
    );

    Tensor input = Tensor::zeros({2, 4});
    Tensor& output = model.forward(input);

    REQUIRE(output.dimensions[0] == 2);
    REQUIRE(output.dimensions[1] == 2);
}

TEST_CASE("the thread pool comes up with at least one worker", "[smoke]") {
    REQUIRE(ThreadPool::size() >= 1);
}
