// Placeholder suite: it exists so the test target, ctest wiring and both thread
// configurations are exercised before the gradient checks land (ticket 007).

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

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

// This is what makes the two ctest invocations differ: the 1-thread run asserts
// a pool of exactly one, the other only that a pool came up at all.
TEST_CASE("the thread pool takes its size from MYGRAD_NUM_THREADS", "[smoke]") {
    const char* requestedThreads = std::getenv("MYGRAD_NUM_THREADS");
    char* firstUnparsedCharacter = nullptr;
    const unsigned long requested = requestedThreads ? std::strtoul(requestedThreads, &firstUnparsedCharacter, 10) : 0;
    const bool requestIsUsable = requestedThreads and *requestedThreads
                                 and *firstUnparsedCharacter == '\0' and requested > 0;

    if (requestIsUsable) {
        REQUIRE(ThreadPool::size() == requested);
    }
    else { // anything unusable, including unset, means automatic sizing
        REQUIRE(ThreadPool::size() >= 1);
    }
}
