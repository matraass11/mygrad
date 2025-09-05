#include <iostream>
#include <vector> 
#include <utility>

#include <filesystem>

#include "mygrad/mygrad.hpp"

#include "src/processData.hpp"
#include "src/modelRunner.hpp"

using namespace mygrad;


int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <train|test>\n";
        return 1;
    }

    Dataset images = loadCatImages(0.9, 0.1);

    const size_t pixelsInImage = 64 * 64 * 3;
    const size_t neurons = 1024;
    const size_t latent = 256;

    const size_t imageSizeBeforeLatent = 256*4*4;

    // Model encoder {
    //     Reshape({1, pixelsInImage}, 0),
    //     LinearLayer(pixelsInImage, neurons),
    //     ReLU(),
    //     LinearLayer(neurons, latent),
    //     ReLU()
    // };


    // Model decoder {
    //     LinearLayer(latent, neurons),
    //     ReLU(),
    //     LinearLayer(neurons, pixelsInImage),
    //     Sigmoid()
    // };


    Model encoder {
        Conv2d(3, 32, 3, 2, 1), // B x 32 x 32 x 32
        ReLU(),
        Conv2d(32, 64, 3, 2, 1), // B x 64 x 16 x 16
        ReLU(),
        Conv2d(64, 128, 3, 2, 1), // B x 128 x 8 x 8
        ReLU(),
        Conv2d(128, 256, 3, 2, 1), // B x 256 x 4 x 4
        ReLU(),
        Reshape({1, imageSizeBeforeLatent}, 0),
        LinearLayer(imageSizeBeforeLatent, latent)
    };

    Model decoder {
        LinearLayer(latent, imageSizeBeforeLatent),
        Reshape({1, 256, 4, 4}, 0),
        Upsample(2),
        Conv2d(256, 128, 3, 1, 1),
        ReLU(),
        Upsample(2),
        Conv2d(128, 64, 3, 1, 1),
        ReLU(),
        Upsample(2),
        Conv2d(64, 32, 3, 1, 1),
        ReLU(),
        Upsample(2),
        Conv2d(32, 3, 3, 1, 1),
        Sigmoid()
    };

    std::string mode = argv[1];

    if (mode == "train") {

        trainModel(encoder, decoder, images.train);
        encoder.save("../encoder.model");
        decoder.save("../decoder.model");

    } else if (mode == "test") {

        encoder.load("../encoder.model");
        decoder.load("../decoder.model");
        testModel(encoder, decoder, images.test, "../testImages");
        std::cout << "test images have been saved to 'cats/testImages/'\n";

    } else {
        std::cerr << "Invalid mode: " << mode << "\n";
        std::cerr << "Expected one of: train, test\n";
        return 1;
    }

}