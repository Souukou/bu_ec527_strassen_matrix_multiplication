# Advanced Matrix Multiplication: An In-Depth Exploration and Optimization of the Strassen Algorithm
Final Project of EC527 High Performance Programming with Multicore and GPUs
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

# measure the performance
./compare_time 
```

