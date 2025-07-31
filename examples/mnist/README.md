**simple mnist classifier using mygrad!**

# to build:

### <ins>requirements</ins>: cmake 3.14+, c++20+

## macos / linux:

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

## windows (visual studio):
```bat
:: Open "x64 Native Tools Command Prompt for VS"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -G "NMake Makefiles"
cmake --build build
```

# to run:

```
cd build 
./mnist <train|test|show>
```
