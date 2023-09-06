cmake_minimum_required(VERSION 3.17)

project(EnergyStorms LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG -pg") # enable profiling and debug printout
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)


# --------------------- get git info -----------------------------------------------
execute_process(COMMAND git log --pretty=format:'%h' -n 1
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

# Check whether we got any revision (which isn't
# always the case, e.g. when someone downloaded a zip
# file from Github instead of a checkout)
if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "N/A")
    set(GIT_DIFF "")
    set(GIT_TAG "N/A")
    set(GIT_BRANCH "N/A")
else()
    execute_process(
        COMMAND bash -c "git diff --quiet --exit-code || echo +"
        OUTPUT_VARIABLE GIT_DIFF)
    execute_process(
        COMMAND git describe --exact-match --tags
        OUTPUT_VARIABLE GIT_TAG ERROR_QUIET)
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        OUTPUT_VARIABLE GIT_BRANCH)

    string(STRIP "${GIT_REV}" GIT_REV)
    string(SUBSTRING "${GIT_REV}" 1 7 GIT_REV)
    string(STRIP "${GIT_DIFF}" GIT_DIFF)
    string(STRIP "${GIT_TAG}" GIT_TAG)
    string(STRIP "${GIT_BRANCH}" GIT_BRANCH)
endif()

set(VERSION "const char* GIT_REV=\"${GIT_REV}${GIT_DIFF}\";
const char* GIT_TAG=\"${GIT_TAG}\";
const char* GIT_BRANCH=\"${GIT_BRANCH}\";")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp VERSION_)
else()
    set(VERSION_ "")
endif()

if (NOT "${VERSION}" STREQUAL "${VERSION_}")
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/include/version.hpp "${VERSION}")
endif()


# Sequential program
add_library(sequential_functions SHARED 
            ${CMAKE_CURRENT_SOURCE_DIR}/src/energy_storms_sequential_functions.cpp)  
target_include_directories(sequential_functions PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)        
add_executable(energy_storms_seq 
                src/energy_storms_sequential.cpp
                )
target_link_libraries(energy_storms_seq PUBLIC
                    sequential_functions)   
target_include_directories(energy_storms_seq PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include) 

# OpenMP program
find_package(OpenMP REQUIRED)

add_library(omp_functions SHARED 
            ${CMAKE_CURRENT_SOURCE_DIR}/src/energy_storms_omp_functions.cpp)

target_include_directories(omp_functions PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include) 

add_executable(energy_storms_omp
                src/energy_storms_omp.cpp)

# Link the energy_storms_omp executable with omp_functions library and OpenMP
target_link_libraries(energy_storms_omp PRIVATE omp_functions OpenMP::OpenMP_CXX)


target_include_directories(energy_storms_omp PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include)

# OMP Test executable
add_executable(omp_test_run_calculation
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/testOMPRunCalculation.cpp
)
target_include_directories(omp_test_run_calculation PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include)  
target_link_libraries(omp_test_run_calculation PRIVATE
                        sequential_functions
                        omp_functions)

include(CTest)

add_test(NAME OMP_basic_test
        COMMAND omp_test_run_calculation 35 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_01_a35_p5_w3 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_01_a35_p7_w2 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_01_a35_p8_w1 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_01_a35_p8_w4
)
add_test(NAME OMP_extreme_positions_1
        COMMAND omp_test_run_calculation 16 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_09_a16-17_p3_w1
)
add_test(NAME OMP_extreme_positions_2
        COMMAND omp_test_run_calculation 17 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_09_a16-17_p3_w1
)
add_test(NAME OMP_test_03
        COMMAND omp_test_run_calculation 20 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_03_a20_p4_w1
)
add_test(NAME OMP_test_04
        COMMAND omp_test_run_calculation 20 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_04_a20_p4_w1
)
add_test(NAME OMP_test_05
        COMMAND omp_test_run_calculation 20 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_05_a20_p4_w1
)
add_test(NAME OMP_test_06
        COMMAND omp_test_run_calculation 20 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_06_a20_p4_w1
)

