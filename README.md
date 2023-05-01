# bu_ec527_strassen_matrix_multiplication

## Build Instructions

### Build

```
mkdir -p build && cd build
cmake ..
make -j
```

### Run

```
# run the main only(only test the strasse)
./strassen_mm

# run all the unit test(source file in ./tests)
make test

# run a single test, e.g. correctness test
./test_strassen
```

