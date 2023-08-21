/*
 * Simplified simulation of high-energy particle storms
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2017/2018
 *
 * Version: 2.0
 *
 * Sequential reference code.
 *
 * (c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include<stdio.h>
#include<stdlib.h>
#include "energy_storms_sequential.hpp"

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i,j,k;

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    SEQUENTIAL::Storm storms[ num_storms ];

    /* 1.2. Read storms information */
    for( i=2; i<argc; i++ ) 
        storms[i-2] = SEQUENTIAL::read_storm_file( argv[i] );

    /* 1.3. Intialize maximum levels to zero */
    float maximum[ num_storms ];
    int positions[ num_storms ];
    for (i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
    }

    /* 2. Begin time measurement */
    double ttotal = SEQUENTIAL::cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */
    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * layer_size );
    if ( layer == NULL) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    SEQUENTIAL::run_calculation(layer, layer_size, storms, num_storms,
                    maximum,
                    positions);

    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    ttotal = SEQUENTIAL::cp_Wtime() - ttotal;

    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
    #ifdef DEBUG 
    SEQUENTIAL::debug_print( layer_size, layer, positions, maximum, num_storms );
    #endif

    /* 7. Results output, used by the Tablon online judge software */
    printf("\n");
    /* 7.1. Total computation time */
    printf("Time: %lf\n", ttotal );
    /* 7.2. Print the maximum levels */
    printf("Result:");
    for (i=0; i<num_storms; i++)
        printf(" %d %f", positions[i], maximum[i] );
    printf("\n");

    /* 8. Free resources */    
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );

    /* 9. Program ended successfully */
    return 0;
}

