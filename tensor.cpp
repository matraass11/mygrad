#include <iostream>
#include "grad.h"
class Sum;
class Product;
class Value;

class Tensor {
public:
    int rows, columns;
    // std::array<Value, rows> weights; 

    Value** values;
    explicit Tensor(int rows, int columns):
    rows(rows), columns(columns)
    {
        values = new Value*[rows*columns];
        for (int i=0; i<rows*columns; i++){
            values[i] = new Value(0);
        }
    }
    
    void gaussianInit(){
        ;
    }

    // void initAllTo(double value){
    //     for (int i; i < rows*columns)
    // }


};