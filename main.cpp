#include <iostream>
#include <array>
#include <fstream>

#include "src/tensor.hpp"
#include "src/model.hpp"
#include "src/loss.hpp"
#include "src/optim.hpp"
#include "src/helper.hpp"

#define MODELS_DIR "../models/"

int main() {

    Tensor images = load_mnist_images("/Users/fedorkurmanov/Desktop/prog/neuralNetCpp/datasets/mnist/train-images-ubyte");
    Tensor imglabels = load_mnist_labels("/Users/fedorkurmanov/Desktop/prog/neuralNetCpp/datasets/mnist/train-labels-ubyte");
    Tensor standartizedImages = standartize(images);
    visualize_image_ascii(images, imglabels, 10);

    const size_t batchSize = 32;

    Model model;

    CrossEntropyLoss loss(10);
    Adam optim(model.parameters, 0.001);
    int n_steps = 2;

    for (int step = 0; step < n_steps; step++) {
        std::vector<size_t> indices = shuffledIndices(imglabels.length);
        dtype avgLoss = 0;
        for (int batch = 0; batch < imglabels.length / batchSize; batch++) {
            std::vector<size_t> batchIndices = slicedIndices(indices, batch*32, 32);
            Tensor batchInputs = retrieveBatchFromData(standartizedImages, batchIndices);
            Tensor batchLabels = retrieveBatchFromLabels(imglabels, batchIndices);

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
        avgLoss /= (imglabels.length / batchSize);
        std::cout << "average loss at " << step << ": " << avgLoss << "\n";
    }


    // for (int i = 0; i < 1001; i++) {
    //     std::memset(logits.grads.get(), 0, logits.length * sizeof(dtype));
    //     dtype l = loss.forward(logits, labels);
    //     loss.backward();
    //     if (i % 100 == 0) {
    //         std::cout << "loss at " << i << ": " << l << "\n";
    //         logits.print();
    //     }
    //     optim.step();
    // }
} 
