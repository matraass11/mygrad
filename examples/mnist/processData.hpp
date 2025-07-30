#pragma once

#include "mygrad/mygrad.hpp"

using namespace mygrad;

Tensor loadMnistImages(const std::string& path);
Tensor loadMnistLabels(const std::string& path); 
void visualizeImage(const Tensor& images, const Tensor& labels, int index); 