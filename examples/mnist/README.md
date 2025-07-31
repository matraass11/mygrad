# to build:

<ins> requirements: </ins> cmake, c++20

## macos / linux:

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## windows (visual studio):
```bat
:: Open "x64 Native Tools Command Prompt for VS"

cmake -B build -DCMAKE_BUILD_TYPE=Release -G "NMake Makefiles"
cmake --build build
```

# to run:

```
cd build 
./mnist <train|test|show>
```