cmake_minimum_required(VERSION 3.14)

project(EnergyStorms LANGUAGES CXX)


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

add_library(mpi_functions SHARED 
            ${CMAKE_CURRENT_SOURCE_DIR}/src/energy_storms_mpi_functions.cpp)  
target_include_directories(mpi_functions PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)        
add_executable(energy_storms_mpi
                src/energy_storms_mpi.cpp
                )
target_link_libraries(energy_storms_seq PUBLIC
                    mpi_functions)   
target_include_directories(energy_storms_seq PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include)




# Sets DEBUG precompiler flag to run visual output correctly
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(energy_storms_seq PRIVATE
                            "-DDEBUG")                     
endif()

