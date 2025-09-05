#pragma once 

#include "mygrad/mygrad.hpp"

void trainModel( mygrad::Model& encoder, mygrad::Model& decoder, mygrad::Tensor& trainingImages );
void testModel( mygrad::Model& encoder, mygrad::Model& decoder, mygrad::Tensor& testImages, const std::string& dirForImages );
void generateImages( mygrad::Model& encoder, mygrad::Model& decoder, size_t latent, const std::string& dirForImages );