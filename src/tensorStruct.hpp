#pragma once

struct Tensor;

template<size_t length>
struct Tensor_s {

    double dataArray[length];
    double gradArray[length];
    Tensor tensor;

    Tensor_s(const std::array<double, length>& dataArray, const std::vector<int>& dimensionsVector) :  
        dataArray{ }, gradArray{ }, 
        tensor(Tensor(this->dataArray, this->gradArray, dimensionsVector)) {
            std::copy(dataArray.begin(), dataArray.end(), this->dataArray);
            check_dimensions();
        }
    
    Tensor_s(const std::vector<int>& dimensionsVector) :  
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
struct TensorMatmulProduct_s : public Tensor_s<length> {
    TensorMatmulProduct_s(const Tensor& leftInputTensor, const Tensor& rightInputTensor) : 
        Tensor_s<length>({leftInputTensor.dimensions[0], rightInputTensor.dimensions[1]}) {}
};

template<size_t length>
struct TensorSum_s : public Tensor_s<length> {
    TensorSum_s(const Tensor& leftInputTensor, const Tensor& rightInputTensor) :
    Tensor_s<length>(leftInputTensor.dimensions) {}
};