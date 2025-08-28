#include <iostream>
#include <filesystem>

#include "modelRunner.hpp"
#include "processData.hpp"
#include "mygrad/mygrad.hpp"

using namespace mygrad;

static const size_t pixelsInImage = 64 * 64 * 3;
static const size_t neurons = 512;
static const size_t latent = 64;


void trainModel() {

    Dataset images = loadCatImages(0.9, 0.1, 0);
    convertTensorToPng(images.train, 100, "../test.png");

    for (size_t i = 0; i < images.train.length; i++) {
        images.train.data[i] /= 255.0; 
    }



    Model encoder {
        Reshape({1, pixelsInImage}, 0),
        LinearLayer(pixelsInImage, neurons),
        ReLU(),
        LinearLayer(neurons, latent * 2), // first 20 columns of output are the means, second 20 columns are the logvariances. 
        // reparameterize simply interprets it that way 
        // and outputs a tensor with values sampled from the distributions of shape [batchSize x latent]
    };

    Reparameterize reparam;


    Model decoder {
        LinearLayer(latent, neurons),
        ReLU(),
        LinearLayer(neurons, pixelsInImage),
        Sigmoid()
    };


    std::vector<Tensor*> parameters = encoder.parameters;

    parameters.insert( parameters.end(), decoder.parameters.begin(), decoder.parameters.end() );
    Adam optim(parameters, 0.000001);

    KLdivWithStandardNormal kldiv;
    MSEloss mse;

    const size_t epochs = 1;
    const size_t batchSize = 64;

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        Tensor& data = images.train;
        const size_t trainSize = data.dimensions[0];

        const std::vector<size_t> indices = shuffledIndices(trainSize);
        dtype avgLoss = 0;
        for (int batch = 0; batch < trainSize / batchSize; batch++) {
            std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
            Tensor batchInputs = retrieveBatchFromData(data, batchIndices);

            
            Tensor& latentDistribution = encoder(batchInputs); 
            reparam.forward(latentDistribution);
            Tensor& outputs = decoder(reparam.outputTensor);

            dtype loss = kldiv(latentDistribution) + mse(outputs, batchInputs); // order is important

            if (batch % 5 == 0) {
                std::cout << "batch - " << batch << ", loss - " << loss << "\n";
            }
            // if (batch == 80) break;

            avgLoss += loss;


            encoder.zeroGrad(), reparam.outputTensor.zeroGrad(), decoder.zeroGrad();
            mse.backward();
            kldiv.backward();
            decoder.backward();
            // reparam.outputTensor.printGrad();
            // break;
            // decoder.layers[1].outputTensor.printGrad();
            reparam.backward();
            encoder.backward();

            optim.step();

            // if (batch == 3) break;

        }
        avgLoss /= (trainSize / batchSize);
    }

    encoder.save("../encoder.model");
    decoder.save("../decoder.model");
}

void generateImages() {
    Reparameterize reparam;

    Model decoder {
        LinearLayer(latent, neurons),
        ReLU(),
        LinearLayer(neurons, pixelsInImage),
        Sigmoid()
    };

    decoder.load("../decoder.model");

    const size_t amountOfImages = 10;
    Tensor normDistWithLogVariance = Tensor::zeros({amountOfImages, latent * 2});
    reparam.forward(normDistWithLogVariance);
    Tensor& output = decoder.forward(reparam.outputTensor);
    for (size_t i = 0; i < output.length; i++) {
        output.data[i] *= 255;
    }
    Reshape reshaper({amountOfImages, 3, 64, 64});
    reshaper.forward(output);
    Tensor& images = reshaper.outputTensor;

    for (size_t i = 0; i < images.dimensions[0]; i++) {
        convertTensorToPng(images, i, std::string("../newCats/cat_" + std::to_string(i) + ".png"));
    }
    
}


// void testModel(Model& model) {
//     std::filesystem::path path = std::filesystem::current_path();

//     Tensor testImages = loadMnistImages(path / "../dataset/test-images-ubyte");
//     Tensor testLabels = loadMnistLabels(path / "../dataset/test-labels-ubyte");
//     Tensor standartizedTestImages = standartize(testImages);

//     std::cout << "test accuracy: " << test(model, standartizedTestImages, testLabels, 1024) << std::endl;
// }