#include <iostream>
#include <array>
#include <filesystem>



#include <vector> // TEMPORARY

#include "mygrad/mygrad.hpp"

#include "src/processData.hpp"
#include "src/modelRunner.hpp"

using namespace mygrad;

int main(int argc, char* argv[]) {

    Conv2d testConv(2, 2, 3, 2, 1);
    testConv.kernels = Tensor(std::vector<dtype>(testConv.kernels.length, 1), testConv.kernels.dimensions);
    Tensor input( {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, {1, 2, 3, 3});

    // testConv.currentInputTensor = &input;
    // testConv.im2col(input, testConv.matrixFormCurrentInput);
    // std::cout << testConv.matrixFormCurrentInput.dimensions;
    // testConv.matrixFormCurrentInput.print();

    // Tensor im2colImplResult = Tensor(
    //     std::vector<dtype>(testConv.outputTensor.data.get(),
    //                        testConv.outputTensor.data.get() + testConv.outputTensor.length), 
    //     testConv.outputTensor.dimensions);

    Tensor& currentResult = testConv.outputTensor;
    testConv.forwardWithIm2col(input);
    std::cout << "the new result:\n";
    testConv.outputTensor.print();
    std::cout << "\n\n";
    
    testConv(input);
    std::cout << "the old result:\n";
    currentResult.print();




    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <train|test|show>\n";
        return 1;
    }
    
    const size_t pixelsInImage = 784;
    const size_t numberOfClasses = 10;
    const size_t neurons = 100;
    const size_t channels = 32;
    const size_t inputsToLinear = std::pow((28 - 5 + 1 - 3 + 1) / 2, 2) * channels;

    Model model (
        // LinearLayer( pixelsInImage, neurons ),
        // ReLU(),
        // LinearLayer( neurons, neurons ),
        // ReLU(),
        // LinearLayer( neurons, numberOfClasses )
        Conv2d(1, channels, 5, 1),
        ReLU(),
        Conv2d(channels, channels, 3, 1),
        ReLU(),
        MaxPool2d(2),
        Reshape( {1, inputsToLinear }, 0),
        LinearLayer( inputsToLinear, neurons ),
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
