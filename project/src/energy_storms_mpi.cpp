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
#include "version.hpp"

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
    bool error = false;
    if(mpi_info.rank == mpi_info.root){
        if (argc<3) {
            fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
            error = true;
        }
    }
    MPI_Bcast(&error, 1, MPI_C_BOOL, mpi_info.root, MPI_COMM_WORLD); //Just to ensure all the processes get the error
    if(error){
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;


    std::vector<MPI_FUNCTIONS::Storm> storms( num_storms );

    /* 1.2. Read storms information */
    MPI_FUNCTIONS::read_storm_files(argc, argv, storms); //Check if reading scales with all processes (i.e. if parallel read is used)

    /* 1.3. Intialize maximum levels to zero */
    std::vector<float> maximum( num_storms, 0.0f );
    std::vector<int> positions( num_storms, 0.0 );
    // Print out build info
    if(mpi_info.rank == mpi_info.root){
        std::cout << "Revision: " << GIT_REV;
        std::cout << ", tag: " << GIT_TAG;
        std::cout << ", branch: " << GIT_BRANCH;
        std::cout << std::endl;
    }
    // Print out MPI info
    if(mpi_info.rank == mpi_info.root){
        std::cout << "Nr mpi processes: " << mpi_info.size;
        std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD); //Barrier to ensure correct time measurement
    if(mpi_info.rank == mpi_info.root){
        /* 2. Begin time measurement */
        ttotal = MPI_FUNCTIONS::cp_Wtime();
    }
    /* START: Do NOT optimize/parallelize the code of the main program above this point */
    /* 3. Allocate memory for the layer and initialize to zero */
    std::vector<float> layer(layer_size, 0.0f);
    // if ( layer == NULL) { //TODO replace with try catch if needed
    //     fprintf(stderr,"Error: Allocating the layer memory\n");
    //     MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    // }
    MPI_FUNCTIONS::run_calculation(layer,
                    storms, 
                    maximum,
                    positions,
                    mpi_info);
    
    /* END: Do NOT optimize/parallelize the code below this point */
    if(mpi_info.rank == mpi_info.root){
        /* 5. End time measurement */
        ttotal = MPI_FUNCTIONS::cp_Wtime() - ttotal; //No barrier needed as only root process holds final result
    
        /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
        #ifdef DEBUG 
        MPI_FUNCTIONS::debug_print(layer, positions, maximum, num_storms );
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
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* 8. Free resources */    
    //Not needed due to use of containers
    
    /* 9. Program ended successfully */
    MPI_Finalize();
    return 0;
}

