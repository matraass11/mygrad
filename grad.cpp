#include <iostream>

class Sum;
class Product;
class Value {
public:
    double data;
    Value *parent1, *parent2;
    double grad = 1;

    Value(double data, Value* parent1=nullptr, Value* parent2=nullptr){
        this->data = data, this->parent1 = parent1, this->parent2 = parent2;
    }

    void propagateBack(){
        // increment parents' grads knowing the grad of the self and how self came to be
        // if self is result of addition and dself/dchild = 2, then we increment parents' grads by 2*1
        // std::cout << "backward undefined" << std::endl;
    }
    Sum operator+(Value other);
    Product operator*(Value other);
}; 

class Sum : public Value {
using Value::Value;
public:
    void propagateBack(){
        parent1->grad += grad*1;
        parent2->grad += grad*1;
        parent1->propagateBack(), parent2->propagateBack();
        // std::cout << "parent1: " << parent1->data << std::endl;
    }
};

Sum Value::operator+(Value other){
    double sum = data+other.data;
    Sum child(sum, parent1=this, parent2=&other);
    return child;
};

class Product : public Value {
using Value::Value;
public:
    void propagateBack(){
        parent1->grad += parent2->data, parent2->grad += parent1->data;
        parent1->propagateBack(), parent2->propagateBack();
        // std::cout << "parent1: " << parent1->data << std::endl;
    }
};

Product Value::operator*(Value other){
    double product = data * other.data;
    Product child(product, parent1=this, parent2=&other);
    return child;
}



int main(){
    Value r1c1(3), r1c2(3), r2c1(4), r2c2(6), n(-1);
    Product pr = r1c1*r1c2;
    // std::cout << pr.parent1 << std::endl;

    Sum det = (r1c1 * r2c2) + (n * r2c1 * r1c2);
    det.propagateBack();
    std::cout << det.parent2 << "buhaha" << std::endl;
    
}