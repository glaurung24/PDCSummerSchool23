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
#include <iostream>
#include "energy_storms_mpi.hpp"
#include <mpi.h>

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {

    if( MPI_Init(&argc, &argv)
        != MPI_SUCCESS){
        std::cerr << "Error while initializing MPI" << std::endl;
        exit(EXIT_FAILURE);
    }
    int size, rank;
    if( MPI_Comm_size(MPI_COMM_WORLD, &size)
        != MPI_SUCCESS){
        std::cerr << "MPI error on line: " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
    if( MPI_Comm_rank(MPI_COMM_WORLD, &rank)
        != MPI_SUCCESS){
        std::cerr << "MPI error on line: " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    MPI_FUNCTIONS::MPIInfo mpi_info;
    mpi_info.size = size;
    mpi_info.rank = rank;
    double ttotal;

    /* 1.1. Read arguments in the root process (assuming parallel read is slow/no parallelization allowed here)*/
    if(mpi_info.rank == mpi_info.root){
        if (argc<3) {
            fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
            exit( EXIT_FAILURE );
        }
    

        int layer_size = atoi( argv[1] );
        int num_storms = argc-2;
        MPI_FUNCTIONS::Storm storms[ num_storms ];

        /* 1.2. Read storms information */
        MPI_FUNCTIONS::read_storm_files(argc, argv, storms, num_storms);

        /* 1.3. Intialize maximum levels to zero */
        float maximum[ num_storms ];
        int positions[ num_storms ];
        for (int i=0; i<num_storms; i++) {
            maximum[i] = 0.0f;
            positions[i] = 0;
        }

        /* 2. Begin time measurement */
        ttotal = MPI_FUNCTIONS::cp_Wtime();
        /* START: Do NOT optimize/parallelize the code of the main program above this point */
        /* 3. Allocate memory for the layer and initialize to zero */
        float *layer = (float *)malloc( sizeof(float) * layer_size );
        if ( layer == NULL) {
            fprintf(stderr,"Error: Allocating the layer memory\n");
            exit( EXIT_FAILURE );
        }
        MPI_FUNCTIONS::run_calculation(layer, layer_size, storms, num_storms,
                        maximum,
                        positions);
        // run_master_process();
    }
    else{
        // run_worker_process();
    }



    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    ttotal = MPI_FUNCTIONS::cp_Wtime() - ttotal;

    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
    #ifdef DEBUG 
    MPI_FUNCTIONS::debug_print( layer_size, layer, positions, maximum, num_storms );
    #endif

    /* 7. Results output, used by the Tablon online judge software */
    printf("\n");
    /* 7.1. Total computation time */
    printf("Time: %lf\n", ttotal );
    /* 7.2. Print the maximum levels */
    printf("Result:");
    for (int i=0; i<num_storms; i++)
        printf(" %d %f", positions[i], maximum[i] );
    printf("\n");

    /* 8. Free resources */    
    for(int i=0; i<argc-2; i++ )
        free( storms[i].posval );

    /* 9. Program ended successfully */
    MPI_Finalize();
    return 0;
}

