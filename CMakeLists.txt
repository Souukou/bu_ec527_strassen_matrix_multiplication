cmake_minimum_required(VERSION 2.8)

project(test_mmm C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1 -fopenmp")

find_library(RT_LIB rt)

add_executable(test_mmm timer.c matrix.c mmm.c test_mmm.c)

target_link_libraries(test_mmm ${RT_LIB})

# Add custom target for code formatting using clang-format
find_program(CLANG_FORMAT "clang-format")

if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_SOURCE_FILES *.c *.h)

  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT}
    -i
    -style=LLVM
    ${ALL_SOURCE_FILES}
    COMMENT "Formatting source code..."
    VERBATIM
  )
endif()