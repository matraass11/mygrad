#pragma once

// _s --> struct

class Tensor;
class TensorMatMulProduct;

template<size_t length>
struct tensor_s_base {
    double dataArray[length];
    double gradArray[length] = {};
};

template<size_t length> // change to size_t
struct tensor_s : tensor_s_base<length> {

    Tensor tensor;

    tensor_s(const std::array<double, length>& dataArray, const std::vector<int>& dimensionsVector) :  
                tensor(Tensor(this->dataArray, this->gradArray, dimensionsVector)) {
                    for (int i=0; i<length; i++)
                        this->dataArray[i] = dataArray[i];
                }
};

template <size_t length>
struct tensor_matmul_product_s : tensor_s_base<length> {

    TensorMatMulProduct tensor;
    tensor_matmul_product_s(Tensor& leftParent, Tensor& rightParent) :
        tensor(this->dataArray, this->gradArray, leftParent, rightParent) {}
};