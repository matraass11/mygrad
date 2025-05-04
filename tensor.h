#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "grad.h"
#include "helper.h"
// class Value;


class Tensor {
    std::vector<int> dimensions;
    Value* values_array;
    int length = 1;

public:
    explicit Tensor(std::vector<int> dimensions, std::vector<double> dataVector) // pass a value creator function instead of a vector?
    : dimensions(dimensions) {
        length = lengthOfTensorFromItsDimensionVector(dimensions);

        if (dataVector.size() != length){
            std::cout << "dataVector size doesn't correspond to the dimensions, exiting" << std::endl;
            exit(1);
        }

        values_array = new Value[length]; //delete[] ? 
                                         //is this a proper way to initialize my object?
        for (int i=0; i<length; i++){
            values_array[i] = Value(dataVector[i]);
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
        outputStream << "]";
    }
    return outputStream;
}

Tensor singleValueTensor(std::vector<int> dimensions, double value){
    std::vector<double> values;
    int length = lengthOfTensorFromItsDimensionVector(dimensions);
    for (int i=0; i<length; i++){
        values.push_back(value);
    }
    return Tensor(dimensions, values);
 }


 Tensor normDistTensor(std::vector<int> dimensions){
    std::vector<double> values;
    int length = lengthOfTensorFromItsDimensionVector(dimensions);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
  
    for (int i=0; i<length; i++){
        values.push_back(distribution(generator));
    }
    Tensor tensor(dimensions, values);
    return tensor;
 }