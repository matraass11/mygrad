# simple autoencoder trained on images of cats using mygrad!

## to build:

### <ins>requirements</ins>: cmake 3.14+, c++20+

### macos / linux:

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=Release 
cmake --build build
```

### windows (visual studio):
```bat
:: Open "x64 Native Tools Command Prompt for VS"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -G "NMake Makefiles"
cmake --build build
```

## to run:

### first, download the [dataset](https://www.kaggle.com/datasets/borhanitrash/cat-dataset) as zip, unpack it into this directory, and rename the directory with the dataset to be catsData: 
```
mv cats catsData
```

### then run the program:
```
cd build 
./cats <train|test>
```
