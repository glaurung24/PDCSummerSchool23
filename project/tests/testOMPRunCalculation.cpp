/*
 * Test for OMP implentation of run_calculation() in high-energy particle storms
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "energy_storms_sequential.hpp"
#include "energy_storms_omp.hpp"

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

    /* 1.2. Read storms information */

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    SEQUENTIAL::Storm storms[ num_storms ];
    SEQUENTIAL::read_storm_files(argc, argv, storms, num_storms);

    OMP_FUNCTIONS::Storm storms_omp[ num_storms ];
    OMP_FUNCTIONS::read_storm_files(argc, argv, storms_omp, num_storms);
    

    // /* 1.2. Read storms information */
    // for (i = 2; i < argc; i++) {
    //     storms_omp[i - 2] = OMP_FUNCTIONS::read_storm_file(argv[i]);
    //     storms[i - 2] = SEQUENTIAL::read_storm_file(argv[i]);
    // }

    /* 1.3. Intialize maximum levels to zero */
    float maximum[ num_storms ];
    int positions[ num_storms ];
    float maximum_omp[ num_storms ];
    int positions_omp[ num_storms ];

    for (int i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
        maximum_omp[i] = 0.0f;
        positions_omp[i] = 0;
    }

    /* START: Do NOT optimize/parallelize the code of the main program above this point */
    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * layer_size );
    float *layer_omp = (float *)malloc( sizeof(float) * layer_size );

    if ( layer == NULL) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    SEQUENTIAL::run_calculation(layer, layer_size, storms, num_storms,
                    maximum,
                    positions);
    OMP_FUNCTIONS::run_calculation(layer_omp, layer_size, storms_omp, num_storms,
                    maximum_omp,
                    positions_omp);

    bool error = false;
    for(int i = 0; i < layer_size; i++){
        if(abs(layer[i] - layer_omp[i]) > abs(layer[i]*EPS) ){
            std::cout << "Error in layer check" << std::endl;
            std::cout << "Expected: " << layer[i] << ", Actual: " << layer_omp[i] << std::endl;
            error = true;
            break;
        }
    }
    for(int i = 0; i < num_storms; i++){
        if(abs(maximum[i] - maximum_omp[i]) > abs(maximum[i]*EPS) ){
            std::cout << "Error in maximum check" << std::endl;
            std::cout << "Expected: " << maximum[i] << ", Actual: " << maximum_omp[i] << std::endl;
            error = true;
            break;
        }
    }
    for(int i = 0; i < num_storms; i++){
        if(positions[i] != positions_omp[i]){
            std::cout << "Error in positions check" << std::endl;
            std::cout << "Expected: " << positions[i] << ", Actual: " << positions_omp[i] << std::endl;
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