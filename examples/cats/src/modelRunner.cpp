#include <iostream>
#include <filesystem>

#include "modelRunner.hpp"
#include "processData.hpp"
#include "mygrad/mygrad.hpp"

using namespace mygrad;


static void trainForOneEpoch ( 
        Model& encoder, Model& decoder, Reparameterize& reparam, 
        Tensor& data, size_t batchSize, size_t epoch,
        Adam& optim, MSEloss& mse, KLdivWithStandardNormal& kldiv );

static std::pair<dtype, dtype> validateModel ( 
        Model& encoder, Model& decoder, Reparameterize& reparam, 
        Tensor& data, size_t batchSize, 
        MSEloss& mse, KLdivWithStandardNormal& kldiv );


void trainModel( Model& encoder, Model& decoder, Dataset& dataset ) {

    for (size_t i = 0; i < dataset.train.length; i++) {
        dataset.train.data[i] /= 255.0; 
    }
    for (size_t i = 0; i < dataset.eval.length; i++) {
        dataset.eval.data[i] /= 255.0; 
    }

    std::vector<Tensor*> parameters = encoder.parameters;
    parameters.insert( parameters.end(), decoder.parameters.begin(), decoder.parameters.end() );

    Reparameterize reparam;
    Adam optim(parameters, 0.001);
    KLdivWithStandardNormal kldiv;
    MSEloss mse("sum");

    const size_t epochs = 50;
    const size_t trainingPatience = 10, LRpatience=5;
    const size_t trainBatchSize = 64, evalBatchSize = 512;

    dtype currentEvalLoss;
    size_t epochsWithoutImprovement = 0;

    for (size_t epoch = 0; epoch < epochs; epoch++) {

        if (epochsWithoutImprovement > trainingPatience) break;

        else if (epochsWithoutImprovement > LRpatience) optim.learningRate /= 10;

        trainForOneEpoch(encoder, decoder, reparam, dataset.train, trainBatchSize, epoch, optim, mse, kldiv);
        auto [msevalue, kldivvalue] = validateModel(encoder, decoder, reparam, dataset.eval, evalBatchSize, mse, kldiv);

        if (msevalue + kldivvalue >= currentEvalLoss) epochsWithoutImprovement++;
        else epochsWithoutImprovement = 0; 
        
        currentEvalLoss = msevalue + kldivvalue;
        std::cout << "epoch " << epoch << " completed. evaluation loss: " << msevalue + kldivvalue << ", (mse - " << msevalue <<  ", kldiv - " << kldivvalue << ")\n";
        encoder.save("../encoder.model");
        decoder.save("../decoder.model");
    }

    std::cout << "training finished.\n";
}


static void trainForOneEpoch ( 
        Model& encoder, Model& decoder, Reparameterize& reparam, 
        Tensor& data, size_t batchSize, size_t epoch,
        Adam& optim, MSEloss& mse, KLdivWithStandardNormal& kldiv ) {

    const size_t trainSize = data.dimensions[0];
    const std::vector<size_t> indices = shuffledIndices(trainSize);
    for (int batch = 0; batch < trainSize / batchSize; batch++) {
        const std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
        Tensor batchInputs = retrieveBatchFromData(data, batchIndices);

        Tensor& latentDistributions = encoder(batchInputs);
        Tensor& encoding = reparam(latentDistributions);
        Tensor& outputs = decoder(encoding);

        dtype msePartOfLoss = mse(outputs, batchInputs), kldivPartOfLoss = kldiv(latentDistributions, std::min(1.0, epoch / 20.0));

        if (batch % 30 == 0) {
            printf("%s%u%s%.4f%s%.4f%s%.4f%s\n", "batch - ", batch, ", loss - ",
                msePartOfLoss + kldivPartOfLoss, ", (mse - ", msePartOfLoss, ", kldiv - ", kldivPartOfLoss, ")");
        }

        encoder.zeroGrad(), reparam.zeroGrad(), decoder.zeroGrad();
        mse.backward(), kldiv.backward(), decoder.backward(), reparam.backward(), encoder.backward();

        optim.step();

    }
}


static std::pair<dtype, dtype> validateModel ( 
        Model& encoder, Model& decoder, Reparameterize& reparam, 
        Tensor& data, size_t batchSize, 
        MSEloss& mse, KLdivWithStandardNormal& kldiv ) {

    const size_t evalSize = data.dimensions[0];

    const std::vector<size_t> indices = shuffledIndices(evalSize);
    
    dtype msePartOfLoss = 0, kldivPartOfLoss = 0;
    for (int batch = 0; batch < evalSize / batchSize; batch++) {
        const std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
        Tensor batchInputs = retrieveBatchFromData(data, batchIndices);

        Tensor& latentDistributions = encoder(batchInputs);
        Tensor& encoding = reparam(latentDistributions);
        Tensor& outputs = decoder(encoding);

        msePartOfLoss += mse(outputs, batchInputs), kldivPartOfLoss += kldiv(latentDistributions, 1); // beta should be one for the validation
    }
    return {msePartOfLoss / (evalSize / batchSize), kldivPartOfLoss / (evalSize / batchSize)};
}


void reconstructImages( Model& encoder, Model& decoder, Tensor& testImages, const std::string& dirForImages) {

    const std::vector<size_t> allShuffledIndices = shuffledIndices(testImages.dimensions[0]);
    const std::vector<size_t> indices = slicedIndices(allShuffledIndices, 0, 10);

    Tensor testBatch = retrieveBatchFromData(testImages, indices);
    Tensor normalizedTestBatch = Tensor::zeros(testBatch.dimensions);

    for (size_t i = 0; i < testBatch.length; i++) {
        normalizedTestBatch.data[i] = testBatch.data[i] / 255.0;
    }

    // Tensor& output = decoder(encoder(normalizedTestBatch)); 
    // AUTOENCODER VARIANT

    Reparameterize reparam;
    Tensor& latentDistribution = encoder(normalizedTestBatch);
    Tensor& output = decoder(reparam(latentDistribution)); 
    // VAE VARIANT

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

void generateImages( Model& encoder, Model& decoder, size_t latent, const std::string& dirForImages ) {
    Reparameterize reparam;
    const size_t amountOfImages = 10;


    // EXPERIMENT FOR SHOWING THAT IMAGE GEN FROM A REGULAR AUTOENCODER WON'T WORK 

    
    // Dataset dataset = loadCatImages(0.9, 0.1);
    // Tensor averageLatents = Tensor::zeros({latent});
    // size_t divisor = 0;

    // const size_t batchSize = 64;
    // const std::vector<size_t> indices = shuffledIndices(dataset.test.dimensions[0]);
    // for (int batch = 0; batch < dataset.test.dimensions[0] / batchSize; batch++) {
    //     const std::vector<size_t> batchIndices = slicedIndices(indices, batch*batchSize, batchSize);
    //     Tensor batchInputs = retrieveBatchFromData(dataset.test, batchIndices);

    //     for (size_t i = 0; i < batchInputs.length; i++) {
    //         batchInputs.data[i] /= 255.0;
    //     }
        
    //     Tensor& latents = encoder(batchInputs);
    //     for (size_t image = 0; image < latents.dimensions[0]; image++) {
    //         for (size_t latentDim = 0; latentDim < latents.dimensions[1]; latentDim++){
    //             averageLatents.data[latentDim] += latents.at({image, latentDim});
    //         }
    //     }

    //     divisor += latents.dimensions[0];
    // }

    // for (size_t i = 0; i < averageLatents.length; i++) {
    //     averageLatents.data[i] /= divisor;
    // }

    // Tensor averageLatentsWithNoise = Tensor::zeros({amountOfImages, latent});
    // for (size_t image = 0; image < amountOfImages; image++) {
    //     for (size_t latentDim = 0; latentDim < latent; latentDim++) {
    //         const std::vector<dtype> noiseToAdd = normDistVector(latent, 0.05);
    //         averageLatentsWithNoise.at({image, latentDim}) = averageLatents.at({latentDim}) + noiseToAdd[latentDim];
    //     }
    // }

    // Tensor& output = decoder(averageLatentsWithNoise);

    // END OF EXPERIMENT CODE
    // Reparameterize reparam;

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
        convertTensorToPng(images, i, std::string(dirForImages + "/cat_" + std::to_string(i) + ".png"));
    }
    
}