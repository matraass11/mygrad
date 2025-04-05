#include <iostream>

class Sum;
class Value {
public:
    double data;
    std::vector<Value*> parents;
    double grad = 1;

    Value(double data, std::vector<Value*> parents = std::vector<Value*>()){
        this->data = data, this->parents = parents;
    }

    void propagateBack(){
        // increment parents' grads knowing the grad of the self and how self came to be
        // if self is result of addition and dself/dchild = 2, then we increment parents' grads by 2*1
    }
    Sum operator+(Value other);
}; 

class Sum : public Value {
using Value::Value;
public:
    void propagateBack(){
        for (auto parent: parents)
            parent->grad += grad*1;
    }
};

// class Product : public Value {
// using Value::Value;
// public:
//     void propagateBack(){
//         for (auto parent: parents)
//             parent->grad += grad*1;
// }
// }

Sum Value::operator+(Value other){
    std::vector<Value*> parents({this, &other});
    double sum = data+other.data;
Sum child(sum, parents);
    return child;
};

//Sum* Value::operator+(Value* other){
//     double sum = data + other->data;
//    Sum number(sum, std::vector<Value*> {this, other});
// }



int main(){
    Value parentNumber1(3), parentNumber2(4);
    Value number(7, std::vector<Value*> {&parentNumber1, &parentNumber2});
    number.propagateBack();
    std::cout << number.data << std::endl;
    for (auto parent : number.parents)
        std::cout << parent->grad << ' ';
    std::cout << std::endl;
Sum sum = parentNumber1 + parentNumber2;
    std::cout << sum.data << std::endl;
}