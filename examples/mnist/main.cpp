#include <iostream>
#include <array>
#include <fstream>
#include <filesystem>

#include "mygrad/mygrad.hpp"

#include "processData.hpp"

using namespace mygrad;

dtype train(Model& model, Tensor& data, Tensor& labels, CrossEntropyLoss& loss, Adam& optim, size_t batchSize) {
    std::vector<size_t> indices = shuffledIndices(labels.length);
    dtype avgLoss = 0;
    for (int batch = 0; batch < labels.length / batchSize; batch++) {
        std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
        Tensor batchInputs = retrieveBatchFromData(data, batchIndices);
        Tensor batchLabels = retrieveBatchFromLabels(labels, batchIndices);

        batchInputs.reshape({batchIndices.size(), 784});

        Tensor& output = model.forward(batchInputs);

        dtype l = loss(output, batchLabels);

        std::cout << "batch - " << batch << ", loss - " << l << "\n";
        avgLoss += l;
        
        loss.backward();
        model.backward();
        optim.step();
        model.zeroGrad();

    }
    avgLoss /= (labels.length / batchSize);
    return avgLoss;
}

dtype test(Model& model, Tensor& data, Tensor& labels, size_t batchSize) {
    std::vector<size_t> indices = shuffledIndices(labels.length);

    int correct = 0;
    for (int batch = 0; batch < labels.length / batchSize; batch++) {
        std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
        Tensor batchInputs = retrieveBatchFromData(data, batchIndices);
        Tensor batchLabels = retrieveBatchFromLabels(labels, batchIndices);

        batchInputs.reshape({batchIndices.size(), 784});

        const Tensor& predictions = model.forward(batchInputs).argmax(1);

        for (int i = 0; i < predictions.length; i++) {
            if (predictions.data[i] == batchLabels.data[i]) {
                correct++;
            }
        }

        std::cout << predictions.dimensions << "correct: " << correct << ", " << "predictions.length: " << predictions.length << "\n"; 

    }
    return correct / (dtype) labels.length;
}


int main() {

    Tensor t ( {10000, 2, 3, 4, 5, 2, 99, 999}, {2, 2, 2});

    std::cout << t.locationIn1dArrayToIndices(3);

    // HANDLE TENSOR CONSTRUCTOR BAD ALLOC WHEN ACCIDENTALLY PASSING DATA AS DIMENSIONS

    t.print();
    t.argmax(2).print();
    t.argmax(1).print();
    t.argmax(0).print();

    // std::filesystem::path path = std::filesystem::current_path();

    // Tensor images = loadMnistImages(path /"../dataset/train-images-ubyte");
    // Tensor labels = loadMnistLabels(path /"../dataset/train-labels-ubyte");
    // Tensor standartizedImages = standartize(images);
    // visualizeImage(images, labels, 10);

    // const size_t inputSize = images.strides[0], outputSize = 10;
    // const size_t n_neurons = 100;

    // Model model (
    //     LinearLayer( inputSize, n_neurons ),
    //     ReLU(),
    //     LinearLayer( n_neurons, n_neurons ),
    //     ReLU(),
    //     LinearLayer( n_neurons, outputSize )
    // );

    // CrossEntropyLoss loss(10);
    // Adam optim(model.parameters, 0.001);
    // int n_steps = 2;
    // const size_t batchSize = 32;

    // for (int step = 0; step < n_steps; step++) {
    //     // dtype avgTrainLoss = train(model, standartizedImages, labels, loss, optim, 32);
    //     std::cout << "test accuracy on step " << step << ": " << test(model, standartizedImages, labels, 1024) << std::endl;
    // }



    // Tensor testImages = loadMnistImages(path / "../datasets/mnist/test-images-ubyte");
    // Tensor testLabels = loadMnistLabels(path / "../datasets/mnist/test-labels-ubyte");
    // Tensor stdTestImages = standartize(images);
    // visualizeImage(testImages, 88);

    // const size_t testBatchSize = 32;
    // for ()
} 
