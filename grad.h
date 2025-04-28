#pragma once

class Sum;
class Product;
class Value {
public:

    double data;
    Value * parent1, *parent2;
    double grad = 0;

    explicit Value() : data(0), parent1(nullptr), parent2(nullptr) {}

    explicit Value(double data, Value* parent1=nullptr, Value* parent2=nullptr) : 
        data(data), parent1(parent1), parent2(parent2) {}

    virtual void propagateFurther(){
        return;
    }

    void backward(){
        grad = 1;
        propagateFurther();
    }

    Sum operator+(Value& other);
    Product operator*(Value& other);
    
    friend std::ostream& operator<< (std::ostream& outputStream, const Value& value);
}; 

std::ostream& operator<< (std::ostream& outputStream, const Value& value){
    outputStream << value.data;
    return outputStream;
}

class Sum : public Value {
using Value::Value;
public:
    void propagateFurther() override {
        parent1->grad += grad*1, parent2->grad += grad*1;
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
        parent1->propagateFurther(), parent2->propagateFurther();
    }
};

Product Value::operator*(Value& other) {
    double product = data * other.data;
    // std::cout << &other << std::endl;
    Product child(product, this, &other);
    return child;
}
