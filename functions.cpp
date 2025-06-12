#include <iostream>
#include "functions.hpp"
#include "tensor.hpp"

mat_mul2d::mat_mul2d(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor) :
    leftInputTensor(leftInputTensor), rightInputTensor(rightInputTensor), outputTensor(outputTensor) {
        checkDimensions();
    }

void mat_mul2d::checkDimensions() {
    if (
        leftInputTensor.dimensions.size() != 2 or
        rightInputTensor.dimensions.size() != 2 or
        outputTensor.dimensions.size() != 2
    ) {
        std::cerr << "dimensionality must be 2, exiting\n";
        exit(1);
    }

    if (
        leftInputTensor.dimensions[1] != rightInputTensor.dimensions[0] or
        leftInputTensor.dimensions[0] != outputTensor.dimensions[0] or
        rightInputTensor.dimensions[1] != outputTensor.dimensions[1]
    ) {
        std::cerr << "wrong dimensions, exiting\n";
        exit(1);
    }
}

void mat_mul2d::forward() {
    for (int row=0; row < outputTensor.dimensions[0]; row++) {
        for (int column=0; column < outputTensor.dimensions[1]; column++) {
            double& currentElement = outputTensor.at({row, column}) = 0;

            for (int dotProductIterator=0; dotProductIterator < leftInputTensor.dimensions[1]; dotProductIterator++) {
                currentElement +=
                    leftInputTensor.at({row, dotProductIterator}) * 
                    rightInputTensor.at({dotProductIterator, column});
            }
        }
    }
}

void mat_mul2d::backward() {
    for (int row=0; row < outputTensor.dimensions[0]; row++) {
        for (int column=0; column < outputTensor.dimensions[1]; column++) {
            double& currentGradPassedDown = outputTensor.gradAt({row, column});
            
            for (int dotProductIterator=0; dotProductIterator < leftInputTensor.dimensions[1]; dotProductIterator++) {
                leftInputTensor.gradAt({row, dotProductIterator}) +=
                    rightInputTensor.at({dotProductIterator, column}) * currentGradPassedDown; 

                rightInputTensor.gradAt({dotProductIterator, column}) += 
                    leftInputTensor.at({row, dotProductIterator}) * currentGradPassedDown; 
            }
        }
    }
}