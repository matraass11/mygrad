#pragma once

#include <iostream>
#include "grad.h"
class Value;

class Tensor {
public:
    int rowsN, columnsN;
    // std::array<Value, rows> weights; 

    Value*** rows;

    explicit Tensor(int rowsN, int columnsN):
    rowsN(rowsN), columnsN(columnsN)
    {
        rows = new Value**[rowsN];
        for (int i=0; i<rowsN; i++){
            rows[i] = new Value*[columnsN];
        }
    }

    // class Row {
    // public:
    //     int columns;
    //     Row(c)
    // }
    
    Value** operator[](int row){
        return values[columns*row + column];
    }

    void gaussianInit(){
        ;
    }

    // void initAllTo(double value){
    //     for (int i; i < rows*columns)
    // }


};