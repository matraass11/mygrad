#include <iostream>
#include <filesystem>

#include "modelRunner.hpp"
#include "processData.hpp"
#include "mygrad/mygrad.hpp"

using namespace mygrad;

static dtype train(Model& model, Tensor& data, Tensor& labels, CrossEntropyLoss& loss, Adam& optim, size_t batchSize);
static dtype test(Model& model, Tensor& data, Tensor& labels, size_t batchSize);


void trainModel(Model& model) {
    std::filesystem::path path = std::filesystem::current_path();

    Tensor images = loadMnistImages(path /"../dataset/train-images-ubyte");
    Tensor labels = loadMnistLabels(path /"../dataset/train-labels-ubyte");
    Tensor standartizedImages = standartize(images);

    CrossEntropyLoss loss;
    Adam optim(model.parameters, 0.001);

    train(model, standartizedImages, labels, loss, optim, 64);
}


void testModel(Model& model) {
    std::filesystem::path path = std::filesystem::current_path();

    Tensor testImages = loadMnistImages(path / "../dataset/test-images-ubyte");
    Tensor testLabels = loadMnistLabels(path / "../dataset/test-labels-ubyte");
    Tensor standartizedTestImages = standartize(testImages);

    std::cout << "test accuracy: " << test(model, standartizedTestImages, testLabels, 1024) << std::endl;
}

void showModel(Model& model) {
    std::filesystem::path path = std::filesystem::current_path();

    Tensor testImages = loadMnistImages(path / "../dataset/test-images-ubyte");
    Tensor testLabels = loadMnistLabels(path / "../dataset/test-labels-ubyte");
    Tensor standartizedTestImages = standartize(testImages);

    std::cout << "choosing random images from test set...\n";
    std::vector<size_t> randomIndices = shuffledIndices(testLabels.length); 
    for (int i = 0; i < 5; i++) {

        std::cout << "\n==============================\n";
        Tensor input = retrieveBatchFromData(standartizedTestImages, {randomIndices[i]});
        input.reshape({1, 784});
        visualizeImage(testImages, randomIndices[i]);
        const Tensor prediction = model(input).argmax(1);
        // std::cout << "\n==============================\n";
        std::cout <<   "        prediction: " << prediction.data[0] << "         \n";
        std::cout <<   "==============================\n";

    }
}


static dtype train(Model& model, Tensor& data, Tensor& labels, CrossEntropyLoss& loss, Adam& optim, size_t batchSize) {
    std::vector<size_t> indices = shuffledIndices(labels.length);
    dtype avgLoss = 0;
    for (int batch = 0; batch < labels.length / batchSize; batch++) {
        std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
        Tensor batchInputs = retrieveBatchFromData(data, batchIndices);
        Tensor batchLabels = retrieveBatchFromLabels(labels, batchIndices);

        batchInputs.reshape({batchIndices.size(), static_cast<size_t>(batchInputs.strides[0])});

        Tensor& output = model.forward(batchInputs);

        dtype l = loss(output, batchLabels);

        if (batch % 5 == 0) {
            std::cout << "batch - " << batch << ", loss - " << l << "\n";
        }
        avgLoss += l;
        
        loss.backward();
        model.backward();
        optim.step();
        model.zeroGrad();


        // FOR SPEEDUP TESTING 
        if (batch == 25) {
            break;
        }

    }
    avgLoss /= (labels.length / batchSize);
    return avgLoss;
}

static dtype test(Model& model, Tensor& data, Tensor& labels, size_t batchSize) {
    std::vector<size_t> indices = shuffledIndices(labels.length);

    int correct = 0;
    int total = 0;
    for (int batch = 0; batch < labels.length / batchSize; batch++) {
        std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
        Tensor batchInputs = retrieveBatchFromData(data, batchIndices);
        Tensor batchLabels = retrieveBatchFromLabels(labels, batchIndices);

        batchInputs.reshape({batchSize, static_cast<size_t>(batchInputs.strides[0])});

        const Tensor& predictions = model.forward(batchInputs).argmax(1);

        for (int i = 0; i < predictions.length; i++) {
            if (predictions.data[i] == batchLabels.data[i]) {
                correct++;
            }
            total++;
        }

    }
    return correct / (dtype) total;
}



