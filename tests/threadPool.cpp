#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

#include "mygrad/threadPool.hpp"

using namespace mygrad;

// This is what makes the two ctest invocations differ: the 1-thread run asserts a
// pool of exactly one, the other only that a pool came up at all.
TEST_CASE("the thread pool takes its size from MYGRAD_NUM_THREADS", "[threadPool]") {
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
