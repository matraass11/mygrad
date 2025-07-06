// #include <iostream>
// #include "functions.hpp"
// #include "tensor.hpp"

// twoInputFunction::twoInputFunction() : {}

// Sum::Sum() : {}

// void Sum::forward() {
//     for (int i = 0; i < leftInputTensor.length; i++){
//         outputTensor.data[i] = leftInputTensor.data[i] + rightInputTensor.data[i];
//     }
// }

// void Sum::backward() {
//     for (int i = 0; i < leftInputTensor.length; i++){
//         leftInputTensor.grads[i] = rightInputTensor.grads[i] = outputTensor.grads[i];
//     }
// }

// void Sum::checkDimensions() {
//     if (
//         leftInputTensor.dimensions != rightInputTensor.dimensions or
//         leftInputTensor.dimensions != outputTensor.dimensions or
//         rightInputTensor.dimensions != outputTensor.dimensions
//     ) {
//         std::cerr << "dimensions must be identical, exiting\n";
//         exit(1);
//     }
// }

// LossFunction::LossFunction() : {}

// MseLoss::MseLoss() : {}

// void MseLoss::checkDimensions() {
//     if (dataTensor.dimensions != modelOutputTensor.dimensions or lossTensor.length != 1) {
//         std::cerr << "dimensions error for MseLoss, exiting\n";
//         exit(1);
//     }
// }

// void MseLoss::forward() {
//     double loss = 0;
//     double error;
//     for (int i = 0; i < dataTensor.length; i++) {
//         error = (dataTensor.data[i] - modelOutputTensor.data[i]);
//         loss += error*error;
//     }
//     loss /= dataTensor.length;
//     lossTensor.data[0] = loss;
// }

// void MseLoss::backward() {
//     // assume the grad of the loss is always 1
//     for (int i = 0; i < dataTensor.length; i++) {
//         double difference = dataTensor.data[i] - modelOutputTensor.data[i];
//         modelOutputTensor.grads[i] += (difference * -2/dataTensor.length);
//     }
// }