#pragma once
#include <vector>

int lengthOfTensorFromItsDimensionVector(std::vector<int> dimensions){
    int length = 1;
    for (int i=0; i<dimensions.size(); i++){
        length *= dimensions[i];
    }
    return length;
}