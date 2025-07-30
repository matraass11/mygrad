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
    int total = 0;
    for (int batch = 0; batch < labels.length / batchSize; batch++) {
        std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
        Tensor batchInputs = retrieveBatchFromData(data, batchIndices);
        Tensor batchLabels = retrieveBatchFromLabels(labels, batchIndices);

        batchInputs.reshape({batchSize, 784});

        const Tensor& predictions = model.forward(batchInputs).argmax(1);

        for (int i = 0; i < predictions.length; i++) {
            if (predictions.data[i] == batchLabels.data[i]) {
                correct++;
            }
            total++;
        }

        // std::cout  << "correct: " << correct << ", " << "total: " << total << "\n"; 

    }
    return correct / (dtype) total;
}


int main() {




    // HANDLE TENSOR CONSTRUCTOR BAD ALLOC WHEN ACCIDENTALLY PASSING DATA AS DIMENSIONS

    // Tensor t ({2, 3, 4, 0, 1, 0, -2, -34, -0.005, 980, 7, 0.1}, {4, 3});
    // t.argmax(1).print();




    std::filesystem::path path = std::filesystem::current_path();

    Tensor images = loadMnistImages(path /"../dataset/train-images-ubyte");
    Tensor labels = loadMnistLabels(path /"../dataset/train-labels-ubyte");
    Tensor standartizedImages = standartize(images);
    visualizeImage(standartizedImages, labels, 10);


    Tensor testImages = loadMnistImages(path / "../dataset/test-images-ubyte");
    Tensor testLabels = loadMnistLabels(path / "../dataset/test-labels-ubyte");
    Tensor standartizedTestImages = standartize(testImages);
    visualizeImage(testImages, testLabels, 88);


    const size_t inputSize = images.strides[0], outputSize = 10;
    std::cout << inputSize << "\n";
    const size_t n_neurons = 100;

    Model model (
        LinearLayer( inputSize, n_neurons ),
        ReLU(),
        LinearLayer( n_neurons, n_neurons ),
        ReLU(),
        LinearLayer( n_neurons, outputSize )
    );

    CrossEntropyLoss loss(10);
    Adam optim(model.parameters, 0.001);
    int n_steps = 1;
    model.load("test_model");

    for (int step = 0; step < n_steps; step++) {
        // std::cout << "average traing loss on step " << step << ": " << train(model, standartizedImages, labels, loss, optim, 128) << std::endl;
        std::cout << "test accuracy on step " << step << ": \n" << test(model, standartizedTestImages, testLabels, 1024) << std::endl;
    }
} 
