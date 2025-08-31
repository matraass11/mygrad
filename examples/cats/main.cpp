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
    const size_t latent = 128;

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
        std::cerr << "Expected one of: train, test, show\n";
        return 1;
    }

}