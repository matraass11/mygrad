#include <memory>
#include <iostream>
#include "tensor.hpp"

Tensor::Tensor( std::vector<uint> dimensionsVector,
                size_t length ) :  
    dataArrayPtr(std::make_unique<double[]>(length)), 
    gradArrayPtr(std::make_unique<double[]>(length)),
    dimensions(dimensionsVector),
    length(lengthFromDimensionsVector(dimensionsVector)) {}
        

Tensor::Tensor( std::vector<double> dataVector, 
                std::vector<uint> dimensionsVector,
                size_t length ) :  
    Tensor(dimensionsVector, length)
    {
        for (int i = 0; i < length; i++){
            dataArrayPtr[i] = dataVector[i];
        }
    }

void Tensor::print(){
    printRecursively(0, 0, length);
    std::cout<<"\n";
}

size_t Tensor::lengthFromDimensionsVector(std::vector<uint>& dimensionsVector){
    size_t length = 1;
    for (int i=0; i<dimensions.size(); i++){
        length *= dimensions[i];
    }
    return length;
}