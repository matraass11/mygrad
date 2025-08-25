#include <iostream>
#include <vector> 
#include <utility>

#include <filesystem>

#include "mygrad/mygrad.hpp"

#include "processData.hpp"


int main() {

    using namespace mygrad;

    Dataset catsData = loadCatImages(0.8, 0.1, 0.1);
    convertTensorToPng(catsData.train, 100, "../testtt.png");

}