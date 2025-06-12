#pragma once

struct Tensor;

template<size_t length>
struct tensor_s {

    double dataArray[length];
    double gradArray[length];
    Tensor tensor;

    tensor_s(const std::array<double, length>& dataArray, const std::vector<int>& dimensionsVector) :  
        dataArray{ }, gradArray{ }, 
        tensor(Tensor(this->dataArray, this->gradArray, dimensionsVector)) {
            std::copy(dataArray.begin(), dataArray.end(), this->dataArray);
            check_dimensions();
        }
    
    tensor_s(const std::vector<int>& dimensionsVector) :  
        dataArray{ }, gradArray{ },
        tensor(Tensor(this->dataArray, this->gradArray, dimensionsVector)) {}

    void check_dimensions() {
        int productOfDimensions = 1;
        for (auto dim : tensor.dimensions){
            productOfDimensions *= dim;
        }
        if (productOfDimensions != length) {
            std::cerr << "dims dont correspond to length\n";
            exit(1);
        }
    }
};

template<size_t length>
struct tensor_matmul_product_s : public tensor_s<length> {
    tensor_matmul_product_s(const Tensor& leftInputTensor, const Tensor& rightInputTensor) : 
        tensor_s<length>({leftInputTensor.dimensions[0], rightInputTensor.dimensions[1]}) {}
};

template<size_t length>
struct tensor_sum_s : public tensor_s<length> {
    tensor_sum_s(const Tensor& leftInputTensor, const Tensor& rightInputTensor) :
    tensor_s<length>(leftInputTensor.dimensions) {}
};