#include <iostream>
#include <fstream>
#include <vector> 
#include <utility>

#include "mygrad/mygrad.hpp"

#include "processData.hpp"


int main() {
    Dataset catsData = loadCatImages(0.8, 0.1, 0.1);
}