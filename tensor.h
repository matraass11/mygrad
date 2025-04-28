#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "grad.h"
// class Value;

class Tensor {
    std::vector<int> dimensions;
    Value* values_array;
    int length = 1;

public:
    explicit Tensor(std::vector<int> dimensions, std::vector<double> dataVector) // pass a value creator function instead of a vector?
    : dimensions(dimensions) {
        for (auto dim: dimensions){
            length *= dim;
        }

        if (dataVector.size() != length){
            std::cout << "dataVector size doesn't correspond to the dimensions, exiting" << std::endl;
        }

        values_array = new Value[length]; //delete[] ? 
                                         //is this a proper way to initialize my object?
        for (int i=0; i<length; i++){
            values_array[i] = Value(dataVector[i]);
        }
    }

    explicit Tensor(std::vector<int> dimensions, double ValueForAll) // pass a value creator function instead of a vector?
    : dimensions(dimensions) {
        for (auto dim: dimensions){
            length *= dim;
        }
        values_array = new Value[length]; //delete[] ? 
                                         //is this a proper way to initialize my object?
        for (int i=0; i<length; i++){
            values_array[i] = Value(ValueForAll);
        }
    }

    explicit Tensor(const Tensor& IndexedInto, int index) // constructor specifically for indexing
    : dimensions(), values_array(), length(IndexedInto.length / IndexedInto.dimensions[0])
    {
        for (int i=1; i<IndexedInto.dimensions.size(); i++){
            dimensions.push_back(IndexedInto.dimensions[i]);
        }
        values_array = IndexedInto.values_array + index * length;
    }
    
    
    Tensor operator[](int index) const {
        Tensor subTensor = Tensor(*this, index);
        return subTensor;
    }

    friend std::ostream& operator<<(std::ostream& outputStream, const Tensor& tensor);
};

std::ostream& operator<<(std::ostream& outputStream, const Tensor& tensor){
    if (tensor.dimensions.size() == 0){
        outputStream << *tensor.values_array;
    }
    else {
        outputStream << "[";
        for (int i=0; i < tensor.dimensions[0]; i++){
            outputStream << tensor[i];
            if (i<tensor.dimensions[0]-1){
                outputStream << ", ";
            }
        }
        //remove trailing whitespace?
        outputStream << "]";
    }
    return outputStream;
}

// Tensor gaussianTensor(std::vector<int> dimensions){
//     std::vector<Value> values;
//     //implement gaussian here
// }

    // void gaussianInit(){
    //     unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //     std::default_random_engine generator(seed);
    //     std::normal_distribution<double> d(0, 1);
    //     for (int)
    //         d(generator)
    // }