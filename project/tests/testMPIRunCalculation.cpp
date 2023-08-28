/*
 * Test for MPI implentation of run_calculation() in high-energy particle storms
 */
#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include "energy_storms_sequential.hpp"
#include "energy_storms_mpi.hpp"

#define EPS 1E-6 //precision of float

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    SEQUENTIAL::Storm storms[ num_storms ];
    

    /* 1.2. Read storms information */
    SEQUENTIAL::read_storm_files(argc, argv, storms, num_storms);
    MPI_FUNCTIONS::Storm storms_mpi[ num_storms ];
    MPI_FUNCTIONS::read_storm_files(argc, argv, storms_mpi, num_storms);
    // for (int i=0; i<num_storms; i++){
    //     storms_mpi[i] = reinterpret_cast<MPI_FUNCTIONS::Storm>( storms[i] );
    // }

    /* 1.3. Intialize maximum levels to zero */
    float maximum[ num_storms ];
    int positions[ num_storms ];
    float maximum_mpi[ num_storms ];
    int positions_mpi[ num_storms ];
    for (int i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
        maximum_mpi[i] = 0.0f;
        positions_mpi[i] = 0;
    }

    /* START: Do NOT optimize/parallelize the code of the main program above this point */
    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * layer_size );
    float *layer_mpi = (float *)malloc( sizeof(float) * layer_size );
    if ( layer == NULL) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    SEQUENTIAL::run_calculation(layer, layer_size, storms, num_storms,
                    maximum,
                    positions);
    MPI_FUNCTIONS::run_calculation(layer_mpi, layer_size, storms_mpi, num_storms,
                    maximum_mpi,
                    positions_mpi);

    bool error = false;
    for(int i = 0; i < layer_size; i++){
        if(abs(layer[i] - layer_mpi[i]) > abs(layer[i]*EPS) ){
            std::cout << "Error in layer check" << std::endl;
            error = true;
            break;
        }
    }
    for(int i = 0; i < num_storms; i++){
        if(abs(maximum[i] - maximum_mpi[i]) > abs(maximum[i]*EPS) ){
            std::cout << "Error in maximum check" << std::endl;
            error = true;
            break;
        }
    }
    for(int i = 0; i < num_storms; i++){
        if(positions[i] != positions_mpi[i]){
            std::cout << "Error in positions check" << std::endl;
            error = true;
            break;
        }
    }


    /* 8. Free resources */    
    for(int i=0; i<argc-2; i++ )
        free( storms[i].posval );
    if(error){
        exit(EXIT_FAILURE);
    }
    /* 9. Program ended successfully */
    return 0;
}