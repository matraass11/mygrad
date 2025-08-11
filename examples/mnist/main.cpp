#include <iostream>
#include <array>
#include <filesystem>

#include "mygrad/mygrad.hpp"

#include "src/processData.hpp"
#include "src/modelRunner.hpp"

using namespace mygrad;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <train|test|show>\n";
        return 1;
    }
    
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


    std::string mode = argv[1];

    if (mode == "train") {
        trainModel(model);
        model.save(std::filesystem::current_path() / "../model");
    } else if (mode == "test") {
        model.load(std::filesystem::current_path() / "../model");
        testModel(model);
    } else if (mode == "show") {
        model.load(std::filesystem::current_path() / "../model");
        showModel(model);
    } else {
        std::cerr << "Invalid mode: " << mode << "\n";
        std::cerr << "Expected one of: train, test, show\n";
        return 1;
    }


    // HANDLE TENSOR CONSTRUCTOR BAD ALLOC WHEN ACCIDENTALLY PASSING DATA AS DIMENSIONS

    // Tensor t ({2, 3, 4, 0, 1, 0, -2, -34, -0.005, 980, 7, 0.1}, {4, 3});
    // t.argmax(1).print();


} 
