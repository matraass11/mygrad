#pragma once 

#include "mygrad/mygrad.hpp"

void trainModel( mygrad::Tensor& trainingImages );
void testModel( mygrad::Tensor& testImages);
void generateImages();