#include <iostream>
#include <fstream>
#include <string> 
#include <cmath>

#include "processData.hpp"

using namespace mygrad;

Tensor loadMnistImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file: " + path);

    int magic = 0, n_images = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic, sizeof(int));
    file.read((char*)&n_images, sizeof(int));
    file.read((char*)&n_rows, sizeof(int));
    file.read((char*)&n_cols, sizeof(int));

    magic = __builtin_bswap32(magic);
    if (magic != 2051) throw std::runtime_error("magic number for images wrong");

    n_images      = __builtin_bswap32(n_images);
    n_rows        = __builtin_bswap32(n_rows);
    n_cols        = __builtin_bswap32(n_cols);

    Tensor images = Tensor::zeros({static_cast<size_t>(n_images), 1, static_cast<size_t>(n_rows), static_cast<size_t>(n_cols)});

    for (size_t i = 0; i < static_cast<size_t>(n_images); ++i) {
        for (size_t r = 0; r < static_cast<size_t>(n_rows); ++r) {
            for (size_t c = 0; c < static_cast<size_t>(n_cols); ++c) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, 1);
                images.at({i, 0, r, c}) = static_cast<dtype>(pixel) / 255.0;
            }
        }
    }

    return images;
}

Tensor loadMnistLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file: " + path);

    int magic = 0, n_labels = 0;
    file.read((char*)&magic, sizeof(int));
    file.read((char*)&n_labels, sizeof(int));

    magic = __builtin_bswap32(magic);
    n_labels = __builtin_bswap32(n_labels);
    
    if (magic != 2049) throw std::runtime_error("magic number for labels wrong");


    Tensor labels = Tensor::zeros({static_cast<size_t> (n_labels) }); // Assuming 1D tensor
    for (size_t i = 0; i < static_cast<size_t>(n_labels); ++i) {
        unsigned char label = 0;
        file.read((char*)&label, 1);
        labels.at({i}) = static_cast<float>(label);
    }

    return labels;
}


void visualizeImage(const Tensor& images, size_t index) {
    const size_t imagesDimensionality = images.dimensions.size();
    if (imagesDimensionality != 3 and imagesDimensionality != 4) throw std::runtime_error("image to visualize must have dimensionality = 3 or 4");

    const size_t rows = images.dimensions[imagesDimensionality - 2];
    const size_t cols = images.dimensions[imagesDimensionality - 1];

    const char* grayRamp = " .:-=+*#%@"; // from light to dark

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dtype pixel;
            if (imagesDimensionality == 4) {
                pixel = images.at({index, 0, r, c});
            }
            else {
                pixel = images.at({index, r, c});
            };
            pixel = std::max(0.0, std::min(1.0, pixel));
            size_t level = (pixel * (strlen(grayRamp) - 1));
            std::cout << grayRamp[level];
        }
        std::cout << "\n";
    }
}

void visualizeImage(const Tensor& images, const Tensor& labels, size_t index) {
    
    std::cout << "Label: " << static_cast<int>(labels.at({index})) << "\n";
    visualizeImage(images, index);
}