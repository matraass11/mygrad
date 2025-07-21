#include <iostream>
#include <fstream>
#include <random>
#include <exception>

#include "helper.hpp"
#include "types.hpp"

static std::random_device dev;
static std::mt19937 generator(dev());

std::vector<dtype> normDistVector(size_t length, dtype standardDeviation = 1) {
    std::normal_distribution<double> normDist{0, standardDeviation};
    std::vector<dtype> v(length);
    for (int i = 0; i < length; i++) {
        v[i] = normDist(generator);
    }

    return v;
}

std::vector<dtype> KaimingWeightsVector(size_t inFeatures, size_t outFeatures) {
    dtype variance = 2 / ( (dtype)inFeatures );
    return normDistVector(inFeatures*outFeatures, sqrt(variance));
}


Tensor load_mnist_images(const std::string& path) {
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

Tensor load_mnist_labels(const std::string& path) {
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

void visualize_image_ascii(const Tensor& images, const Tensor& labels, int index) {
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

Tensor retrieveBatchFromData(const Tensor& dataTensor, const std::vector<size_t>& indices) {
    std::vector<size_t> batchDimensions = dataTensor.dimensions;
    batchDimensions[0] = indices.size();
    Tensor batchTensor(batchDimensions);
    const size_t subTensorSize = dataTensor.strides[0];
    for (int i=0; i < indices.size(); i++) {
        std::memcpy(batchTensor.data.get() + i*subTensorSize,
                    dataTensor.data.get() + indices[i]*subTensorSize, 
                    subTensorSize*sizeof(dtype));
    }
    return batchTensor;
}

Tensor retrieveBatchFromLabels(const Tensor& labelsTensor, const std::vector<size_t>& indices) {

    Tensor batchTensor( {indices.size() } );
    for (int i=0; i < indices.size(); i++) {
        batchTensor.data[i] = labelsTensor.data[indices[i]];
    }
    return batchTensor;
}