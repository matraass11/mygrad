#include <iostream>
#include <array>
#include <fstream>
#include <filesystem>

#include "src/tensor.hpp"
#include "src/model.hpp"
#include "src/loss.hpp"
#include "src/optim.hpp"
#include "src/helper.hpp"

#define MODELS_DIR "../models/"

int main() {

    std::filesystem::path path = std::filesystem::current_path();

    Tensor images = loadMnistImages(path /"../datasets/mnist/train-images-ubyte");
    Tensor labels = loadMnistLabels(path /"../datasets/mnist/train-labels-ubyte");
    Tensor standartizedImages = standartize(images);
    visualizeImage(images, labels, 10);

    const size_t batchSize = 32;

    Model model;

    CrossEntropyLoss loss(10);
    Adam optim(model.parameters, 0.001);
    int n_steps = 2;

    for (int step = 0; step < n_steps; step++) {
        std::vector<size_t> indices = shuffledIndices(labels.length);
        dtype avgLoss = 0;
        for (int batch = 0; batch < labels.length / batchSize; batch++) {
            std::vector<size_t> batchIndices = slicedIndices(indices, batch*32, 32);
            Tensor batchInputs = retrieveBatchFromData(standartizedImages, batchIndices);
            Tensor batchLabels = retrieveBatchFromLabels(labels, batchIndices);

            batchInputs.reshape({batchIndices.size(), 784});

            Tensor& output = model.forward(batchInputs);

            dtype l = loss(output, batchLabels);

            // std::cout << "batch - " << batch << ", loss - " << l << "\n";
            avgLoss += l;
            
            loss.backward();
            model.backward();
            optim.step();
            model.zeroGrad();

        }
        avgLoss /= (labels.length / batchSize);
        std::cout << "average loss at " << step << ": " << avgLoss << "\n";
    }



    // Tensor testImages = loadMnistImages(path / "../datasets/mnist/test-images-ubyte");
    // Tensor testLabels = loadMnistLabels(path / "../datasets/mnist/test-labels-ubyte");
    // Tensor stdTestImages = standartize(images);
    // visualizeImage(testImages, 88);

    // const size_t testBatchSize = 32;
    // for ()
} 
