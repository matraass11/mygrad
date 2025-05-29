#pragma once
#include <memory>

class Tensor {
protected:
    std::unique_ptr<double[]> dataArrayPtr;
    std::unique_ptr<double[]> gradArrayPtr;
public:
    const std::vector<uint> dimensions;
    const size_t length;
protected:
    Tensor* parent1;
    Tensor* parent2;

public:

    Tensor( std::vector<uint> dimensionsVector,
            size_t length );
        

    Tensor( std::vector<double> dataVector, 
            std::vector<uint> dimensionsVector,
            size_t length );

    void print();

protected:
    size_t lengthFromDimensionsVector(std::vector<uint>& dimensionsVector);

    void printRecursively(uint start, uint dimension, uint volumeOfPreviousDimension) const {
        uint increment = volumeOfPreviousDimension / dimensions[dimension];
        uint end = start + increment + 1;
        std::cout << "[";
        for (int i=start; i < end; i+=increment){
            if (increment==1){
                std::cout << dataArrayPtr[i];
            }
            else {
                printRecursively(i, dimension+1, increment);
            }
            if (i != end-1){
                std::cout << ", ";
            }
        }
        std::cout << "]";
    }
};