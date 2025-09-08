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

    const size_t epochs = 75;
    const size_t trainingPatience = 10, LRpatience=5;
    const size_t trainBatchSize = 64, evalBatchSize = 512;

    dtype lowestEvalLoss = 999999;
    size_t epochsWithoutImprovement = 0;

    for (size_t epoch = 0; epoch < epochs; epoch++) {

        if (epochsWithoutImprovement > trainingPatience) break;

        else if (epochsWithoutImprovement > LRpatience) optim.learningRate /= 10;

        trainForOneEpoch(encoder, decoder, reparam, dataset.train, trainBatchSize, epoch, optim, mse, kldiv);
        auto [msevalue, kldivvalue] = validateModel(encoder, decoder, reparam, dataset.eval, evalBatchSize, mse, kldiv);

        if (msevalue + kldivvalue >= lowestEvalLoss) epochsWithoutImprovement++;
        else {
            epochsWithoutImprovement = 0; 
            lowestEvalLoss = msevalue + kldivvalue;
        }
        
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


void reconstructImages( Model& encoder, Model& decoder, Tensor& testImages, const std::string& dirForImages ) {

    const std::vector<size_t> allShuffledIndices = shuffledIndices(testImages.dimensions[0]);
    const std::vector<size_t> indices = slicedIndices(allShuffledIndices, 0, 10);

    Tensor testBatch = retrieveBatchFromData(testImages, indices);
    Tensor normalizedTestBatch = Tensor::zeros(testBatch.dimensions);

    for (size_t i = 0; i < testBatch.length; i++) {
        normalizedTestBatch.data[i] = testBatch.data[i] / 255.0;
    }

    Reparameterize reparam;
    Tensor& latentDistribution = encoder(normalizedTestBatch);
    Tensor& reconstructedImages = decoder(reparam(latentDistribution));

    for (size_t i = 0; i < reconstructedImages.length; i++) {
        reconstructedImages.data[i] *= 255;
    }

    std::filesystem::create_directory( dirForImages );

    for (size_t i = 0; i < 10; i++) {
        convertTensorToPng(testBatch, i, dirForImages + "/orig_" + std::to_string(i) + ".png");
        convertTensorToPng(reconstructedImages, i, dirForImages + "/reconstructed_" + std::to_string(i) + ".png");
    }
}

void generateImages( Model& encoder, Model& decoder, size_t latent, const std::string& dirForImages, size_t amountOfImages ) {
    
    Reparameterize reparam;

    Tensor normDistWithLogVariance = Tensor::zeros({amountOfImages, latent * 2});
    reparam.forward(normDistWithLogVariance);
    Tensor& images = decoder.forward(reparam.outputTensor);

    for (size_t i = 0; i < images.length; i++) {
        images.data[i] *= 255;
    }

    std::filesystem::create_directory( dirForImages );

    for (size_t i = 0; i < images.dimensions[0]; i++) {
        convertTensorToPng(images, i, std::string(dirForImages + "/cat_" + std::to_string(i) + ".png"));
    }
    
}