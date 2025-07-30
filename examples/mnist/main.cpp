#include <iostream>
#include <array>
#include <fstream>
#include <filesystem>

#include "mygrad/mygrad.hpp"

#include "processData.hpp"
#include "trainAndTest.hpp"

using namespace mygrad;

int main() {
    
    const size_t pixelsInImage = 784;
    const size_t numberOfClasses = 10;
    const size_t neurons = 100;

    Model model (
        LinearLayer( pixelsInImage, neurons ),
        ReLU(),
        LinearLayer( neurons, neurons ),
        ReLU(),
        LinearLayer( neurons, numberOfClasses )
    );

    if (true) { // command line args
        trainModel(model);
        model.save(std::filesystem::current_path() / "../model");
    }

    model.load("model");
    testModel(model, true);


    // HANDLE TENSOR CONSTRUCTOR BAD ALLOC WHEN ACCIDENTALLY PASSING DATA AS DIMENSIONS

    // Tensor t ({2, 3, 4, 0, 1, 0, -2, -34, -0.005, 980, 7, 0.1}, {4, 3});
    // t.argmax(1).print();


} 
