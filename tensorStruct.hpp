#pragma once

class Tensor;

template<int length> // change to size_t
struct tensorStruct {
    double dataArray[length];
    double gradArray[length] = {};
    Tensor tensor;

    tensorStruct(const std::array<double, length>& dataArray, const std::vector<int>& dimensionsVector) :  
                tensor(Tensor(this->dataArray, this->gradArray, dimensionsVector)) {
                    for (int i=0; i<length; i++)
                        this->dataArray[i] = dataArray[i];
                }
};