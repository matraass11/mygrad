#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <numeric>

#include "mygrad/smallArray.hpp"
#include "mygrad/tensor.hpp"

using namespace mygrad;

// end() used to return &data[size_ - 1], one short of the real end, which would
// have made any range-for over a shape silently drop the last dimension. Nothing
// iterated a SmallArray at the time, so the bug sat there latent.

TEST_CASE("SmallArray iterates over all of its elements", "[smallArray]") {
    TensorDims dimensions = {2, 3, 4};

    REQUIRE(std::distance(dimensions.begin(), dimensions.end()) == 3);

    size_t product = 1;
    for (size_t dimension : dimensions) { product *= dimension; }
    REQUIRE(product == 24);

    REQUIRE(std::accumulate(dimensions.begin(), dimensions.end(), size_t(0)) == 9);
}


TEST_CASE("a one-element SmallArray is iterable", "[smallArray]") {
    const TensorDims dimensions = {7}; // the old end() ran backwards from begin() here

    REQUIRE(std::distance(dimensions.begin(), dimensions.end()) == 1);
    REQUIRE(*dimensions.begin() == 7);
}


TEST_CASE("SmallArray::at rejects out of range indices", "[smallArray]") {
    TensorDims dimensions = {2, 3};

    REQUIRE(dimensions.at(1) == 3);
    REQUIRE_THROWS_AS(dimensions.at(2), std::out_of_range);          // past the size, inside the capacity
    REQUIRE_THROWS_AS(dimensions.at(MAX_TENSOR_DIMENSIONALITY), std::out_of_range);
}
