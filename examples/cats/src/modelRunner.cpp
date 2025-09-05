#include <iostream>
#include <filesystem>

#include "modelRunner.hpp"
#include "processData.hpp"
#include "mygrad/mygrad.hpp"

using namespace mygrad;


void trainModel( Model& encoder, Model& decoder, Tensor& trainingImages ) {

    for (size_t i = 0; i < trainingImages.length; i++) {
        trainingImages.data[i] /= 255.0; 
    }


    std::vector<Tensor*> parameters = encoder.parameters;
    parameters.insert( parameters.end(), decoder.parameters.begin(), decoder.parameters.end() );

    Adam optim(parameters, 0.001);

    // KLdivWithStandardNormal kldiv;
    MSEloss mse;

    const size_t epochs = 50;
    const size_t batchSize = 64;

    for (size_t epoch = 0; epoch < epochs; epoch++) {

        if (epoch == 10) optim.learningRate /= 10;

        else if (epoch == 30) optim.learningRate /= 2;

        Tensor& data = trainingImages;
        const size_t trainSize = data.dimensions[0];

        const std::vector<size_t> indices = shuffledIndices(trainSize);
        for (int batch = 0; batch < trainSize / batchSize; batch++) {
            const std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
            Tensor batchInputs = retrieveBatchFromData(data, batchIndices);

            Tensor& outputs = decoder(encoder(batchInputs));
            dtype loss = mse(outputs, batchInputs);

            if (batch % 25 == 0) {
                printf("%s%u%s%.4f\n", "batch - ", batch, ", loss - ", loss);
            }

            encoder.zeroGrad(), decoder.zeroGrad();
            mse.backward(), decoder.backward(), encoder.backward();

            optim.step();

        }
        encoder.save("../encoder.model");
        decoder.save("../decoder.model");
    }


}

void testModel( Model& encoder, Model& decoder, Tensor& testImages, const std::string& dirForImages) {

    const std::vector<size_t> allShuffledIndices = shuffledIndices(testImages.dimensions[0]);
    const std::vector<size_t> indices = slicedIndices(allShuffledIndices, 0, 10);

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

    std::filesystem::create_directory( dirForImages );

    for (size_t i = 0; i < 10; i++) {
        convertTensorToPng(testBatch, i, dirForImages + "/orig_" + std::to_string(i) + ".png");
        convertTensorToPng(reconstructedImages, i, dirForImages + "/reconstructed_" + std::to_string(i) + ".png");
    }
}

// void generateImages() {
//     Reparameterize reparam;

//     Model decoder {
//         LinearLayer(latent, neurons),
//         ReLU(),
//         LinearLayer(neurons, pixelsInImage),
//         Sigmoid()
//     };

//     decoder.load("../decoder.model");

//     const size_t amountOfImages = 10;
//     Tensor normDistWithLogVariance = Tensor::zeros({amountOfImages, latent * 2});
//     reparam.forward(normDistWithLogVariance);
//     Tensor& output = decoder.forward(reparam.outputTensor);
//     for (size_t i = 0; i < output.length; i++) {
//         output.data[i] *= 255;
//     }
//     Reshape reshaper({amountOfImages, 3, 64, 64});
//     reshaper.forward(output);
//     Tensor& images = reshaper.outputTensor;

//     for (size_t i = 0; i < images.dimensions[0]; i++) {
//         convertTensorToPng(images, i, std::string("../newCats/cat_" + std::to_string(i) + ".png"));
//     }
    
// }