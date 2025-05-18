#pragma once

#include <iostream>


template <size_t length, uint dimensionality>
class TensorBase {    
private:
    std::array<double, length> dataArray;
    std::array<double, length> gradArray;
    const std::array<uint, dimensionality> dimensions; 
    const std::array<uint, dimensionality> strides; 
    const TensorBase *parent1, *parent2;


public:
    TensorBase(std::array<double, length> dataArray, std::array<uint, dimensionality> dimensions) 
    : dataArray(dataArray), gradArray({ }), dimensions(dimensions), strides(stridesFromDimensions(dimensions)) {
        uint productOfDimensions = 1;
        for (auto dim : dimensions){
            productOfDimensions *= dim;
        }
        if (productOfDimensions != length){
            std::cerr << "error: dimensions don't correspond to the length, exiting" << std::endl; 
            exit(1);
        }
    }

    void print() const {
        printRecursively(0, 0);
        std::cout << "\n";
    }

private:

    const std::array<uint, dimensionality> stridesFromDimensions(const std::array<uint, dimensionality>& dimensions){
        std::array<uint, dimensionality> strides;
        strides[0] = length/dimensions[0];
        for (int i=1; i < dimensionality; i++){
            strides[i] = strides[i-1]/dimensions[i];
        }
        return strides;
    }

    
    void printRecursively(uint start, uint dimension) const {
        uint increment = strides[dimension];
        uint end = start + increment + 1;
        std::cout << "[";
        for (int i=start; i < end; i+=increment){
            if (increment==1){
                std::cout << dataArray[i];
            }
            else {
                printRecursively(i, dimension+1);
            }
            if (i != end-1){
                std::cout << ", ";
            }
        }
        std::cout << "]";
    }
    
    // TensorBase matMul2d(const TensorBase& other){
    //     if (!(dimensionality == 2 && other.dimensionality == 2)){
    //         std::cerr << "error: matMul2d only for 2d, exiting\n";
    //         exit(1);
    //     }
    //     if (dimensions[1] != other.dimensions[0]){
    //         std::cerr << "error: invalid shapes, exiting" << std::endl; 
    //         exit(1);
    //     }

    //     std::array<uint, 2> productDimensions = {dimensions[0], other.dimensions[1]};
    //     size_t productLength = dimensions[0] * other.dimensions[1];
    //     TensorMatrixProduct<productLength, 2> product(resultArray, dimensions)
    // }
    // virtual void propagateFurther();
};


    
template <size_t length, uint dimensionality>
class TensorMatrixProduct : public TensorBase <length, dimensionality> {
public:
    TensorMatrixProduct(std::array<double, length> dataArray) 
    : TensorBase<length, dimensionality>(dataArray) {}
    
    // using TensorBase<length>::print;
    // void propagateFurther() override {
        //     print();
        // }
        
        
};







    //     if (dimensionIndex > dimensionality - 1) {
    //         std::cerr << "error: invalid dimension, exiting" << std::endl;
    //         exit(1);
    //     }
    
    //     uint stride = strides[dimensionIndex+1];
    //     uint lastElement = indexOfFirstElement+lengthOfDimension;
    //     std::cout << "[";
    
    //     for (int i=indexOfFirstElement; i < lastElement; i=i+stride){
    //         if (dimensionIndex == dimensionality-2){
    //             std::cout << dataArray[i];
    //         }
    //         else{
    //             printOneDimension(dimensionIndex+1, i, lengthOfDimension/dimensions[dimensionIndex]);
    //         }
    //         if (i != lastElement-1){
    //         std::cout << ", ";
    //     }    
    //     std::cout << "]";
    // }
    
    // }