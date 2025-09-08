#include <iostream>
#include <vector> 
#include <utility>

#include <filesystem>

#include "mygrad/mygrad.hpp"

#include "src/processData.hpp"
#include "src/modelRunner.hpp"
#include "src/cli.hpp"

using namespace mygrad;

#define DEFAULT_AMOUNT_OF_IMAGES 10

int main(int argc, char* argv[]) {
    

    CLIOptions options;
    try {
        options = parseArguments(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }


    Dataset images = loadCatImages(0.9, 0.1);

    const size_t latent = 128;
    
    const size_t pixelsInImage = 64 * 64 * 3;
    const size_t imageSizeBeforeLatent = 256*4*4;

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
        LinearLayer(imageSizeBeforeLatent, latent * 2)
    };

    Model decoder {
        LinearLayer(latent, imageSizeBeforeLatent),
        Reshape({1, 256, 4, 4}, 0),
        Upsample(2), Conv2d(256, 128, 3, 1, 1), ReLU(),
        Upsample(2), Conv2d(128, 64, 3, 1, 1), ReLU(),
        Upsample(2), Conv2d(64, 32, 3, 1, 1), ReLU(),
        Upsample(2), Conv2d(32, 3, 3, 1, 1),
        Sigmoid()
    };


    if (options.mode == "train") {

        trainModel(encoder, decoder, images);
        encoder.save("../newly_trained_encoder.model");
        decoder.save("../newly_trained_decoder.model");

    } else if (options.mode == "reconstruct") {

        encoder.load("../pretrained_encoder.model");
        decoder.load("../pretrained_decoder.model");
        reconstructImages(encoder, decoder, images.eval, "../testImages");
        std::cout << "original and reconstructed images have been saved to 'cats/testImages/'\n";

    } else if (options.mode == "generate") {

        size_t amountOfImages = options.amountOfImages.value_or(DEFAULT_AMOUNT_OF_IMAGES);

        decoder.load("../pretrained_decoder.model");
        generateImages(decoder, latent, "../newCats", amountOfImages);
        std::cout << "new images have been saved to 'cats/newCats/'\n";
    }

}