#include <iostream>
#include <filesystem>

#include "modelRunner.hpp"
#include "processData.hpp"
#include "mygrad/mygrad.hpp"

using namespace mygrad;

static const size_t pixelsInImage = 64 * 64 * 3;
static const size_t neurons = 1024;
static const size_t latent = 128;


void trainModel( Tensor& trainingImages ) {

    for (size_t i = 0; i < trainingImages.length; i++) {
        trainingImages.data[i] /= 255.0; 
    }



    Model encoder {
        Reshape({1, pixelsInImage}, 0),
        LinearLayer(pixelsInImage, neurons),
        ReLU(),
        LinearLayer(neurons, latent), // first 20 columns of output are the means, second 20 columns are the logvariances. 
        // reparameterize simply interprets it that way 
        // and outputs a tensor with values sampled from the distributions of shape [batchSize x latent]

        ReLU(),
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
    Adam optim(parameters, 0.0001);

    KLdivWithStandardNormal kldiv;
    MSEloss mse;

    const size_t epochs = 2;
    const size_t batchSize = 64;

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        Tensor& data = trainingImages;
        const size_t trainSize = data.dimensions[0];

        const std::vector<size_t> indices = shuffledIndices(trainSize);
        dtype avgLoss = 0;
        for (int batch = 0; batch < trainSize / batchSize; batch++) {
            std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
            Tensor batchInputs = retrieveBatchFromData(data, batchIndices);

            
            // Tensor& latentDistribution = encoder(batchInputs); 
            // reparam.forward(latentDistribution);
            // Tensor& outputs = decoder(reparam.outputTensor);
            // dtype loss = kldiv(latentDistribution) + mse(outputs, batchInputs); // order is important


            Tensor& outputs = decoder(encoder(batchInputs));
            dtype loss = mse(outputs, batchInputs);

            if (batch % 5 == 0) {
                std::cout << "batch - " << batch << ", loss - " << loss << "\n";
            }
            // if (batch == 80) break;

            avgLoss += loss;


            // encoder.zeroGrad(), reparam.outputTensor.zeroGrad(), decoder.zeroGrad();
            // mse.backward(), kldiv.backward();
            // decoder.backward(), reparam.backward(), encoder.backward();

            encoder.zeroGrad(), decoder.zeroGrad();
            mse.backward(), decoder.backward(), encoder.backward();

            optim.step();

        }
        avgLoss /= (trainSize / batchSize);
    }

    encoder.save("../encoder.model");
    decoder.save("../decoder.model");


}

void testModel( Tensor& testImages ) {


    Model encoder {
        Reshape({1, pixelsInImage}, 0),
        LinearLayer(pixelsInImage, neurons),
        ReLU(),
        LinearLayer(neurons, latent), 
        ReLU()
    };

    Model decoder {
        LinearLayer(latent, neurons),
        ReLU(),
        LinearLayer(neurons, pixelsInImage),
        Sigmoid()
    };

    encoder.load("../encoder.model");
    decoder.load("../decoder.model");

    const std::vector<size_t> indices = shuffledIndices(testImages.dimensions[0]);
    Tensor testBatch = retrieveBatchFromData(testImages, indices);
    Tensor normalizedTestBatch = Tensor::zeros(testBatch.dimensions);

    for (size_t i = 0; i < testBatch.length; i++) {
        normalizedTestBatch.data[i] = testBatch.data[i] / 255.0;
    }

    Tensor& output = decoder(encoder(normalizedTestBatch));
    Reshape reshaper({1, 3, 64, 64}, 0);
    Tensor& reconstructedImages = reshaper(output);

    for (size_t i = 0; i < reconstructedImages.length; i++) {
        reconstructedImages.data[i] *= 255;
    }

    for (size_t i = 0; i < 10; i++) {
        convertTensorToPng(testBatch, i, "../test/orig_" + std::to_string(i) + ".png");
        convertTensorToPng(reconstructedImages, i, "../test/reconstructed_" + std::to_string(i) + ".png");
    }
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


void testModel(Model& model) {

    Model decoder {
        LinearLayer(latent, neurons),
        ReLU(),
        LinearLayer(neurons, pixelsInImage),
        Sigmoid()
    };

    decoder.load("../decoder.model");
}