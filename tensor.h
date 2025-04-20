#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "grad.h"
// class Value;

class Tensor {
public:
    std::vector<int> dimensions;
    Value* values_array;
    int length = 1;

    explicit Tensor(std::vector<int> dimensions)
    : dimensions(dimensions) {
        for (auto i: dimensions){
            length *= dimensions[i];
        }
        values_array = new Value[length]; //delete[] ? 
                                         //is this a proper way to initialize my object?
    }

    explicit Tensor(Tensor& IndexedInto, int index) // constructor specifically for indexing
    : length(IndexedInto.length / IndexedInto.dimensions[0])
    {
        for (int i=1; i<IndexedInto.length; i++){
            dimensions[i-1] = IndexedInto.dimensions[i];
        }
        values_array = IndexedInto.values_array + index * length * sizeof(Value);
    }
    
    
    Tensor operator[](int index){
        if (this->dimensions.size() > 1){ 
            Tensor subTensor = Tensor(*this, index);
            return subTensor;
        }
    }
    
    // void gaussianInit(){
    //     unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //     std::default_random_engine generator(seed);
    //     std::normal_distribution<double> d(0, 1);
    //     for (int)
    //         d(generator)
    // }

    // void initAllTo(double value){
    //     for (int i; i < rows*columns)
    // }


};