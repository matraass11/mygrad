#include <iostream>
#include <fstream>
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

    Tensor images({static_cast<size_t>(n_images), static_cast<size_t>(n_rows), static_cast<size_t>(n_cols)});

    for (int i = 0; i < n_images; ++i) {
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, 1);
                images.at({i, r, c}) = static_cast<dtype>(pixel) / 255.0;
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


    Tensor labels({static_cast<size_t> (n_labels) }); // Assuming 1D tensor
    for (int i = 0; i < n_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, 1);
        labels.at({i}) = static_cast<float>(label);
    }

    return labels;
}

void visualizeImage(const Tensor& images, const Tensor& labels, int index) {
    const int rows = images.dimensions[1];
    const int cols = images.dimensions[2];

    const char* grayRamp = " .:-=+*#%@"; // from light to dark
    
    std::cout << "Label: " << static_cast<int>(labels.at({index})) << "\n";

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dtype pixel = images.at({index, r, c});
            pixel = std::max(0.0, std::min(1.0, pixel));
            int level = static_cast<int>(pixel * (strlen(grayRamp) - 1));
            std::cout << grayRamp[level];
        }
        std::cout << "\n";
    }
}