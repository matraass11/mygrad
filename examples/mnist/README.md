**simple mnist classifier using mygrad!**

# to build:

### <ins>requirements</ins>: cmake 3.14+, c++20+

## macos / linux:

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=Release 
cmake --build build
```

## windows (visual studio):
```bat
:: Open "x64 Native Tools Command Prompt for VS"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G "NMake Makefiles"
cmake --build build
```

# assets:

a fresh clone carries neither the trained weights nor the dataset — both live on the [v0.1.0 release](https://github.com/matraass11/mygrad/releases/tag/v0.1.0) and are fetched on demand, checksums verified:

```bash
cmake --build build --target fetch-mnist
```

that is `fetch-mnist-model` (6 MB, needed by `test` and `show`) and `fetch-mnist-dataset` (55 MB, needed by everything) together; run either on its own if you only want one. `train` needs both.

# to run:

```
cd build 
./mnist <train|test|show>
```
