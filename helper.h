#pragma once
#include <vector>

int lengthOfTensorFromItsDimensionVector(std::vector<int> dimensions){
    int length = 1;
    for (int i=0; i<dimensions.size(); i++){
        length *= dimensions[i];
    }
    return length;
}

std::vector<double> normDistVector(int length){
    std::vector<double> normallyDistributedVector;

    std::random_device seedGen;
    std::mt19937 generator(seedGen());
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    for (int i=0; i<length; i++){
        normallyDistributedVector.push_back(distribution(generator));
    }
    return normallyDistributedVector;
}

std::vector<double> OneValueVector(double value, int length){
    std::vector<double> values;
    for (int i=0; i<length; i++){
        values.push_back(value);
    }
    return values;
}