#include <iostream>
#include <vector> 
#include <utility>

#include <filesystem>

#include "mygrad/mygrad.hpp"

#include "src/processData.hpp"
#include "src/modelRunner.hpp"


int main() {

    using namespace mygrad;

    Dataset images = loadCatImages(0.9, 0.1);
    trainModel(images.train);
    testModel(images.test);

}