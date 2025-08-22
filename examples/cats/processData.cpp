#include <filesystem>
#include <stdexcept>
#include <string>
#include <future>
#include <cmath>
// #include <thread>

#include "processData.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/image_write.h"

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


static Tensor tensorWithImageData(const std::vector<size_t>& indices) {
    const size_t expectedSize = 64, expectedChannels = 3;
    Tensor dataTensor = Tensor::zeros( {indices.size(), expectedChannels, expectedSize, expectedSize});

    const std::string pathToDir = std::filesystem::current_path() / "../catsData/Data/"; 

    auto worker = [&](size_t start, size_t end) {
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
        std::cout << "i'm done!" << std::endl;
    };

    const size_t threads_n = std::thread::hardware_concurrency();
    const size_t chunkSize = std::ceil(indices.size() / threads_n);
    std::vector<std::future<void>> futures(threads_n);
    for (int thread = 0; thread < threads_n; thread++) {
        const size_t start = chunkSize * thread, end = std::min(start+chunkSize, indices.size()); 
        futures[thread] = std::async(std::launch::async, worker, start, end);
    }

    for (auto& future: futures) future.get();

    convertTensorToPng(dataTensor, 123, "../test.png");

    return dataTensor;
}


Dataset loadCatImages(float trainSplit, float evalSplit, float testSplit) {

    const size_t length = 29843;  //29843
    const size_t trainBoundary = length * trainSplit;
    const size_t evalBoundary = length*(trainSplit + evalSplit);
    const size_t testBoundary = length*(trainSplit + evalSplit + testSplit);

    if (testBoundary != length) {
        throw std::runtime_error("train+eval+test must equal one. equals: " + std::to_string(trainSplit + evalSplit + testSplit));
    }

    std::vector<size_t> indices = shuffledIndices(length);
    
    return Dataset ( 
        tensorWithImageData(slicedIndices(indices, 0, trainBoundary - 1)),  
        tensorWithImageData(slicedIndices(indices, trainBoundary, evalBoundary - 1)),
        tensorWithImageData(slicedIndices(indices, evalBoundary, length - 1))
    );

}