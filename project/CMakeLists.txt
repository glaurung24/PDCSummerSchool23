cmake_minimum_required(VERSION 3.18)

project(EnergyStorms LANGUAGES CXX CUDA)

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

#mpi program
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

add_library(mpi_functions SHARED 
            ${CMAKE_CURRENT_SOURCE_DIR}/src/energy_storms_mpi_functions.cpp)  
target_include_directories(mpi_functions PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)        
add_executable(energy_storms_mpi
                src/energy_storms_mpi.cpp
                )
target_link_libraries(energy_storms_mpi PUBLIC
                    mpi_functions MPI::MPI_CXX)   
target_include_directories(energy_storms_mpi PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include)

#cuda program
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
                set(CMAKE_CUDA_STANDARD 11)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()
        add_library(cuda_functions SHARED 
                ${CMAKE_CURRENT_SOURCE_DIR}/src/energy_storms_cuda_functions.cu)
        set_target_properties(cuda_functions PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_ARCHITECTURES native)
        target_include_directories(cuda_functions PRIVATE
                        ${CMAKE_CURRENT_SOURCE_DIR}/include
                        CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)    
        add_executable(energy_storms_cuda
                        src/energy_storms_cuda.cpp
                        )
        target_link_libraries(energy_storms_cuda PUBLIC
                        cuda_functions MPI::MPI_CXX)   
        target_include_directories(energy_storms_cuda PRIVATE
                        ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()




#Set up testing

include(CTest)
add_executable(mpi_test_run_calculation
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/testMPIRunCalculation.cpp
)
target_include_directories(mpi_test_run_calculation PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)  
target_link_libraries(mpi_test_run_calculation PRIVATE
                            sequential_functions
                            mpi_functions
)
# enable testing functionality
# enable_testing()
add_test(NAME mpi_test_run_calculation
        COMMAND mpi_test_run_calculation 100 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/test_01_a35_p5_w3
)
