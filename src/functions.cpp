#include <iostream>
#include "functions.hpp"
#include "tensor.hpp"

twoInputFunction::twoInputFunction(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor) : 
    leftInputTensor(leftInputTensor), rightInputTensor(rightInputTensor), outputTensor(outputTensor) {
        // checkDimensions();
    }

MatMul2d::MatMul2d(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor) :
    twoInputFunction(leftInputTensor, rightInputTensor, outputTensor) {}

void MatMul2d::checkDimensions() {
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

void MatMul2d::forward() {
    for (int row=0; row < outputTensor.dimensions[0]; row++) {
        for (int column=0; column < outputTensor.dimensions[1]; column++) {
            double& currentElement = outputTensor.at({row, column}) = 0;

            for (int dotProductIterator=0; dotProductIterator < leftInputTensor.dimensions[1]; dotProductIterator++) {
                currentElement +=
                    leftInputTensor.at({row, dotProductIterator}) * rightInputTensor.at({dotProductIterator, column});
            }
        }
    }
}

void MatMul2d::backward() {
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


Sum::Sum(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor) : 
    twoInputFunction(leftInputTensor, rightInputTensor, outputTensor) {}

void Sum::forward() {
    for (int i = 0; i < leftInputTensor.length; i++){
        outputTensor.dataArrayPtr[i] = leftInputTensor.dataArrayPtr[i] + rightInputTensor.dataArrayPtr[i];
    }
}

void Sum::backward() {
    for (int i = 0; i < leftInputTensor.length; i++){
        leftInputTensor.gradArrayPtr[i] = rightInputTensor.gradArrayPtr[i] = outputTensor.gradArrayPtr[i];
    }
}

void Sum::checkDimensions() {
    if (
        leftInputTensor.dimensions != rightInputTensor.dimensions or
        leftInputTensor.dimensions != outputTensor.dimensions or
        rightInputTensor.dimensions != outputTensor.dimensions
    ) {
        std::cerr << "dimensions must be identical, exiting\n";
        exit(1);
    }
}

LossFunction::LossFunction(Tensor& dataTensor, Tensor& modelOutputTensor, Tensor& lossTensor) : 
    dataTensor(dataTensor), modelOutputTensor(modelOutputTensor), lossTensor(lossTensor) {}

MseLoss::MseLoss(Tensor& dataTensor, Tensor& modelOutputTensor, Tensor& lossTensor) :
    LossFunction(dataTensor, modelOutputTensor, lossTensor) {}

void MseLoss::checkDimensions() {
    if (dataTensor.dimensions != modelOutputTensor.dimensions or lossTensor.length != 1) {
        std::cerr << "dimensions error for MseLoss, exiting\n";
        exit(1);
    }
}

void MseLoss::forward() {
    double loss = 0;
    double error;
    for (int i = 0; i < dataTensor.length; i++) {
        error = (dataTensor.dataArrayPtr[i] - modelOutputTensor.dataArrayPtr[i]);
        loss += error*error;
    }
    loss /= dataTensor.length;
    lossTensor.dataArrayPtr[0] = loss;
}

void MseLoss::backward() {
    // assume the grad of the loss is always 1
    for (int i = 0; i < dataTensor.length; i++) {
        double difference = dataTensor.dataArrayPtr[i] - modelOutputTensor.dataArrayPtr[i];
        modelOutputTensor.gradArrayPtr[i] += (difference * -2/dataTensor.length);
    }
}