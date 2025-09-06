#include <filesystem>
#include <stdexcept>
#include <string>
#include <cmath>

#include "processData.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb/image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb/image_write.h"

#include "mygrad/mygrad.hpp"

void visualizeImageChannel(const Tensor& images, size_t index, size_t channel) {
    const size_t rows = images.dimensions[2];
    const size_t cols = images.dimensions[3];

    const char* grayRamp = " .:-=+*#%@"; // from light to dark

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dtype pixel = images.at({index, channel, r, c}) / 255;
            // pixel = std::max(0.0, std::min(1.0, pixel));
            if (pixel > 1 or pixel < 0) throw std::runtime_error("boooo");
            size_t level = static_cast<size_t>(pixel * (strlen(grayRamp) - 1));
            std::cout << grayRamp[level];
        }
        std::cout << "\n";
    }
}

void convertTensorToPng(const Tensor& imgTensor, size_t index, const std::string& filename) {

    if (imgTensor.dimensions.size() != 4) throw std::runtime_error("expected 4d tensor");
    
    // arrange the data as [rows, columns, channels]

    const size_t channels = imgTensor.dimensions[1];
    const size_t rows = imgTensor.dimensions[2];
    const size_t columns = imgTensor.dimensions[3];

    unsigned char* data = new unsigned char[channels*rows*columns];

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < columns; col++) {
            for (size_t channel = 0; channel < channels; channel++) {
                size_t locInData = row * columns * channels + col*channels + channel;
                data[locInData] = imgTensor.at({index, channel, row, col});
            }
        }
    }

    stbi_write_png(filename.c_str(), columns, rows, channels, data, columns*channels);

    delete[] data;
}


static Tensor tensorWithCatData(const std::vector<size_t>& indices) {
    const size_t expectedSize = 64, expectedChannels = 3;
    Tensor dataTensor = Tensor::zeros( {indices.size(), expectedChannels, expectedSize, expectedSize});

    const std::string pathToDir = std::filesystem::current_path() / "../catsData/Data/"; 

    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil(indices.size() / (double) threads_n);
    for (size_t t=0; t < threads_n; t++) {
        size_t start = chunkSize * t, end = std::min(start+chunkSize, indices.size()); 
        ThreadPool::push([&, start, end] {
            for (size_t i = start; i < end; i++) {
        
                int columns, rows, channels;
                std::string filename = pathToDir + "cat_" + std::to_string( indices[i] ) + ".png";
                unsigned char* data = stbi_load(filename.c_str(), &columns, &rows, &channels, 3);  
                // array laid out contiguously as [rows, columns, channels]
        
                if (!data) throw std::runtime_error("failed to load image at " + filename + "\nrun the program from the build directory");
        
                if (columns != expectedSize or rows != expectedSize or channels != expectedChannels) {
                    throw std::runtime_error("data appears to have been read incorrectly, dimensions do not match");
                }
                
        
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t row = 0; row < rows; row++) {
                        for (size_t col = 0; col < columns; col++) {
                            size_t locInData = row*columns*channels + col*channels + channel;
                            dataTensor.at({i, channel, row, col}) = data[locInData];
                        } 
                    }
                }
                free(data);

            }
        });
    }

    ThreadPool::waitUntilDone();

    return dataTensor;
}


Dataset loadCatImages(float trainSplit, float evalSplit) {

    const size_t length = 29843;  //29843
    const size_t trainBoundary = length * trainSplit;
    const size_t evalBoundary = length * (trainSplit + evalSplit);

    if (evalBoundary != length) {
        throw std::runtime_error("train+eval must equal one. equals: " + std::to_string(trainSplit + evalSplit));
    }

    std::vector<size_t> indices = shuffledIndices(length);
    
    return Dataset ( 
        tensorWithCatData(slicedIndices(indices, 0, trainBoundary - 1)),  
        tensorWithCatData(slicedIndices(indices, trainBoundary, length - 1))
    );

}