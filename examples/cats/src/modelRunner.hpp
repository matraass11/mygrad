#pragma once 

#include "mygrad/mygrad.hpp"
#include "processData.hpp"

void trainModel( mygrad::Model& encoder, mygrad::Model& decoder, Dataset& dataset );
void reconstructImages( mygrad::Model& encoder, mygrad::Model& decoder, mygrad::Tensor& testImages, const std::string& dirForImages);
void generateImages( mygrad::Model& encoder, mygrad::Model& decoder, size_t latent, const std::string& dirForImages );