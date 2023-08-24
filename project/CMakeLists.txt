cmake_minimum_required(VERSION 3.14)

project(EnergyStorms LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG -pg") # enable profiling and debug printout
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


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
