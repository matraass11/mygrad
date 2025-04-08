#include <iostream>

class Sum;
class Product;
class Value {
public:
    double data;
    Value * parent1, *parent2;
    double grad = 0;

    Value(double data, Value* parent1=nullptr, Value* parent2=nullptr) : 
        data(data), parent1(parent1), parent2(parent2) {}

    virtual void propagateFurther(){
        std::cout << this->data <<" backward undefined" << std::endl;
    }

    void backward(){
        grad = 1;
        propagateFurther();
    }

    Sum operator+(Value& other);
    Product operator*(Value& other);
}; 

class Sum : public Value {
using Value::Value;
public:
    void propagateFurther() override {
        parent1->grad += grad*1, parent2->grad += grad*1;
        std::cout << this->data << " propagated successfully" << std::endl;
        parent1->propagateFurther(), parent2->propagateFurther();
    }
};

Sum Value::operator+(Value& other){
    double sum = data+other.data;
    Sum child(sum, this, &other);
    return child;
};

class Product : public Value {
using Value::Value;
public:
    void propagateFurther() override {
        parent1->grad += grad * parent2->data, parent2->grad += grad * parent1->data;
        std::cout << this->data << " propagated successfully" << std::endl;
        parent1->propagateFurther(), parent2->propagateFurther();
    }
};

Product Value::operator*(Value& other) {
    double product = data * other.data;
    // std::cout << &other << std::endl;
    Product child(product, this, &other);
    return child;
}

int main(){
    Value r1c1(3), r1c2(5), r2c1(4), r2c2(6), n(-1);
    Product pr1 = r1c1*r2c2, pr2 = r2c1 * r1c2 * n;
    Product mul = (r1c1 + r2c2) * r1c2 * r2c1; 
    //reason why this works and mul2 doesn't is 
    // that the operator only expects the second arg to be an lvalue (i think)
    Sum mul2 = n + r1c1 * r2c2 * r1c2 * r2c1;
    // Sum mul = n + pr1;
    mul.backward();

    std::cout << r1c1.grad << std::endl;
    // det.propagateFurther();
    // std::cout << det.parent1 << "buhaha" << std::endl;
    
}