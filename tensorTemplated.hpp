#pragma once

#include <iostream>

template<size_t length, uint dimensionality>
class TensorMatrixProduct;

template <size_t length, uint dimensionality>
class TensorBase {    
protected:
    std::array<double, length> dataArray;
    std::array<double, length> gradArray;
    
public:
    const std::array<uint, dimensionality> dimensions; 
    const std::array<uint, dimensionality> strides; 
    
protected:
    const TensorBase *parent1, *parent2;

public:

    TensorBase(const std::array<double, length>& dataArray, const std::array<uint, dimensionality>& dimensions) 
    : dataArray(dataArray), gradArray({ }), dimensions(dimensions), strides(stridesFromDimensions(dimensions)) {
        uint productOfDimensions = 1;
        for (auto dim : dimensions){
            productOfDimensions *= dim;
        }
        if (productOfDimensions != length){
            std::cerr << "error: dimensions don't correspond to the length, exiting" << std::endl; 
            exit(1);
        }
        // all this checking every time we construct a tensor?? maybe make a fast constructor
    }
    
    void print() const {
        printRecursively(0, 0);
        std::cout << "\n";
    }

    
    double at(const std::array<uint, dimensionality>& indices){
        return dataArray[indicesToLocationIn1dArray(indices)];
    }

TensorMatrixProduct matMul2d(const TensorBase& other) const {
    // if (!(dimensionality == 2 && other.dimensionality == 2)) {
    //     std::cerr << "error: matMul2d only for 2d, exiting\n";
    //     exit(1);
    // }
    if (dimensions[1] != other.dimensions[0]) {
        std::cerr << "error: invalid shapes, exiting" << std::endl; 
        exit(1);
    }

    std::array<uint, 2> productDimensions = {dimensions[0], other.dimensions[1]};
    const size_t productLength = dimensions[0] * other.dimensions[1];
    std::array<double, productLength> productDataArray{ };

    for (int row=0; row < productDimensions[0]; row++) {
        for (int column=0; column < productDimensions[1]; column++) {
            uint locationOfElementInProductDataArray = row * productDimensions[0] + column;
            for (int dotProductIterator=0; dotProductIterator < dimensions[1]; dotProductIterator++) {
                productDataArray[locationOfElementInProductDataArray] += at({row, dotProductIterator}) * other.at({dotProductIterator, column});
            }
        }
    }

    TensorMatrixProduct<productLength, 2> product(productDataArray, dimensions, this, &other);
    return product;
}


protected:

    TensorBase(const std::array<double, length>& dataArray, const std::array<uint, dimensionality>& dimensions, 
               TensorBase* parent1, TensorBase* parent2) : TensorBase(dataArray, dimensions), 
               parent1(parent1), parent2(parent2) {}
    
    const std::array<uint, dimensionality> stridesFromDimensions(const std::array<uint, dimensionality>& dimensions) const {
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
    
    uint indicesToLocationIn1dArray(const std::array<uint, dimensionality>& indices){
        uint locationOfElementInDataArray = 0;
        for (int i=0; i < dimensionality; i++){
            locationOfElementInDataArray += indices[i] * strides[i];
        }
        if (locationOfElementInDataArray > length){
            std::cerr << "error: invalid indices, exiting";
            exit(1);
        }
        return locationOfElementInDataArray;
    }


    // void propagateFurther();
};


    
template <size_t length, uint dimensionality>
class TensorMatrixProduct : public TensorBase <length, dimensionality> {
public:
    TensorMatrixProduct(const std::array<double, length>& dataArray, const std::array<uint, dimensionality>& dimensions, 
                        const TensorBase* parent1, const TensorBase* parent2) 
    : TensorBase<length, dimensionality>(dataArray, dimensions, parent1, parent2) {}
    
    // using TensorBase<length>::print;
    // void propagateFurther() override {
        //     print();
        // }
        
        
};