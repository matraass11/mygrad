#include <iostream>
#include "functions.hpp"
#include "tensor.hpp"

mat_mul2d::mat_mul2d(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor) :
    leftInputTensor_p(&leftInputTensor), rightInputTensor_p(&rightInputTensor), outputTensor_p(&outputTensor) {
        checkDimensions();
    }

void mat_mul2d::checkDimensions() {
    if (
        leftInputTensor_p->dimensions.size() != 2 or
        rightInputTensor_p->dimensions.size() != 2 or
        outputTensor_p->dimensions.size() != 2
    ) {
        std::cerr << "dimensionality must be 2, exiting\n";
        exit(1);
    }

    if (
        leftInputTensor_p->dimensions[1] != rightInputTensor_p->dimensions[0] or
        leftInputTensor_p->dimensions[0] != outputTensor_p->dimensions[0] or
        rightInputTensor_p->dimensions[1] != outputTensor_p->dimensions[1]
    ) {
        std::cerr << "wrong dimensions, exiting\n";
        exit(1);
    }
}

void mat_mul2d::forward() {
    for (int row=0; row < outputTensor_p->dimensions[0]; row++) {
        for (int column=0; column < outputTensor_p->dimensions[1]; column++) {
            uint locationInOutputTensor = outputTensor_p->indicesToLocationIn1dArray({row, column});
            outputTensor_p->setDataAt(locationInOutputTensor, 0);

            for (int dotProductIterator=0; dotProductIterator < leftInputTensor_p->dimensions[1]; dotProductIterator++) {
                outputTensor_p->incrementDataAt(locationInOutputTensor, 
                    leftInputTensor_p->at({row, dotProductIterator}) * rightInputTensor_p->at({dotProductIterator, column})
            }
        }
    }
}

void mat_mul2d::backward() {
    for (int row=0; row < outputTensor_p->dimensions[0]; row++) {
        for (int column=0; column < outputTensor_p->dimensions[1]; column++) {
            uint locationInOutputTensor = outputTensor_p->indicesToLocationIn1dArray({row, column});
            
            for (int dotProductIterator=0; dotProductIterator < leftInputTensor_p->dimensions[1]; dotProductIterator++) {
                leftInputTensor_p->incrementGradAt({row, dotProductIterator}, 
                    rightInputTensor_p->at({dotProductIterator, column}) * outputTensor_p->gradAt(locationInOutputTensor); 

                rightInputTensor_p->incrementGradAt({dotProductIterator, column}, 
                    leftParent->at({row, dotProductIterator}) * gradArray[locationOfElementInProductTensor]); 
            }
        }
    }
}