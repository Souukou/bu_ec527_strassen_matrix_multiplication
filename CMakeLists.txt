cmake_minimum_required(VERSION 2.8)

project(strassen_mm C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1 -fopenmp")

find_library(RT_LIB rt)

# Add the source files
file(GLOB SOURCES "src/*.c")

# Set the executable
add_executable(strassen_mm ${SOURCES})

target_link_libraries(strassen_mm ${RT_LIB})

# Add CTest support
enable_testing()
include(CTest)

# Add the test files
file(GLOB_RECURSE TEST_SOURCE_FILES tests/*.c)

# Get all source files from the src directory, except the main file
file(GLOB_RECURSE SRC_FILES src/*.c)
list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c)

# Create test executables and link with the appropriate source files
foreach(test_file ${TEST_SOURCE_FILES})
  get_filename_component(test_name ${test_file} NAME_WE)

  add_executable(${test_name} ${test_file} ${SRC_FILES})
  target_include_directories(${test_name} PRIVATE src)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Add custom target for code formatting using clang-format
find_program(CLANG_FORMAT "clang-format")

if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_SOURCE_FILES src/*.c src/*.h tests/*.c tests/*.h)
  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT}
    -i
    -style=LLVM
    ${ALL_SOURCE_FILES}
    COMMENT "Formatting source code with LLVM style..."
    VERBATIM
  )
endif()
