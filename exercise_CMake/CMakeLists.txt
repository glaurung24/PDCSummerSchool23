cmake_minimum_required(VERSION 3.14)

project(example LANGUAGES CXX)

include(CMakePrintHelpers)

add_executable(hello hello.cpp)

add_library(greeting
	SHARED
	greeting.cpp
	greeting.hpp
)

target_link_libraries(hello PRIVATE greeting)

# define executable and its source file
add_executable(example example.cpp)

foreach(key
  IN ITEMS
    NUMBER_OF_LOGICAL_CORES
    NUMBER_OF_PHYSICAL_CORES
    IS_64BIT
    HAS_SSE2
    OS_NAME
    OS_RELEASE
  )
  # query the item ${key} and save its value in the variable _${key}
  cmake_host_system_information(RESULT _${key} QUERY ${key})
  cmake_print_variables(_${key})
endforeach()


# this is here because the config.h will be generated in PROJECT_BINARY_DIR
target_include_directories(example
  PRIVATE
    ${PROJECT_BINARY_DIR}
  )

configure_file(config-template.h config.h @ONLY)

add_executable(exampleMP  MPexample.cpp)

find_package(OpenMP REQUIRED COMPONENTS CXX)

target_link_libraries(exampleMP PRIVATE OpenMP::OpenMP_CXX)
