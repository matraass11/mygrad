#include <iostream>
#include <vector> 
#include <utility>

#include <filesystem>

#include "mygrad/mygrad.hpp"

#include "processData.hpp"


int main() {

    using namespace mygrad;

    // Dataset catsData = loadCatImages(0.8, 0.1, 0.1);
    // convertTensorToPng(catsData.train, 100, "../test.png");

    const size_t pixelsInImage = 64 * 64 * 3;
    const size_t neurons = 128;
    const size_t latent = 20;

    Tensor distribution ( {0, 0.5, 1, 1.5, -1.5, -1, -0.5, 0}, {4, 2} );
    Reparameterize rep;
    rep.forward(distribution);
    rep.outputTensor.print();

    for (size_t i = 0; i < rep.outputTensor.length; i++) {
        rep.outputTensor.grads[i] = 1;
    }
    rep.backward();
    distribution.printGrad();



    Model encoder {
        LinearLayer(pixelsInImage, neurons),
        ReLU(),
        // LinearLayer(128, 20) + LinearLayer(128, 20) --> get the means and the variances for each sample. 
        // might as well put the reparameterization here too. something like
        // DownsampleAndReparameterize(neurons, latent)

        // OR

        LinearLayer(neurons, latent * 2), // first 20 columns of output are the means, second 20 columns are the logvariances. 
        // reparameterize simply interprets it that way 
        // and outputs a tensor with values sampled from the distributions of shape [batchSize x latent]
    };

    // Reparameterize reparam (latent)

    Model decoder {
        LinearLayer(latent, neurons),
        ReLU(),
        LinearLayer(neurons, pixelsInImage)
    };
    
    // KLdiv kldiv;
    // MSE mse;
    // 
    // Tensor& latentDistribution = encoder(inputs); 
    // Tensor& samples = reparam(latentDistribution)
    // Tensor& outputs = decoder(samples)
    // loss = kldiv(latentDistribution, {0, 1}) + mse(inputs, outputs)


    // encoder.zeroGrad(), reparam.zeroGrad(), decoder.zeroGrad()
    // mse.backward()
    // decoder.backward()
    // reparam.backward()
    // encoder.backward()
    // kldiv.backward()


}