#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include <mpi.h>
#include <algorithm>
#include <iterator>
#include <vector>


#include "energy_storms_mpi.hpp"


namespace MPI_FUNCTIONS{

/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

void read_storm_files(int argc,
                    char* argv[], 
                    std::vector<Storm>& storms
                    ){
    for(int i=2; i<argc; i++ ) 
        storms[i-2] = read_storm_file( argv[i] );
}

void run_calculation(std::vector<float>& layer, 
                std::vector<Storm>& storms,
                std::vector<float>& maximum,
                std::vector<int>& positions,
                MPIInfo& mpi_info){



    for(int k=0; k<layer.size(); k++ ){
        layer[k] = 0.0f;
    }
    std::vector<float> layer_sum;
    if(mpi_info.rank == mpi_info.root){
        layer_sum.reserve(layer.size());
    }
    
    /* 4. Storms simulation */
    for(int i=0; i<storms.size(); i++) {

        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        for(int j=mpi_info.rank; j<storms[i].size; j += mpi_info.size ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position */
            int position = storms[i].posval[j*2];

            /* For each cell in the layer */
            for(int k=0; k<layer.size(); k++ ) {
                /* Update the energy value for the cell */
                update( layer, k, position, energy);
            }
        }


        MPI_Reduce(layer.data(), layer_sum.data(), layer.size(), MPI_FLOAT, MPI_SUM, mpi_info.root, MPI_COMM_WORLD);

        if(mpi_info.rank == mpi_info.root){
            layer.swap(layer_sum); //Move data back to layer of root processs
            
            energy_relaxation(layer, AVERAGING_WINDOW_SIZE);

            /* 4.3. Locate the maximum value in the layer, and its position */
            find_local_maximum(layer, maximum[i], positions[i]);
        }
        else{
            for(int k=0; k<layer.size(); k++ ) layer[k] = 0.0f; //Reset layer in other processes
        }
    }
}

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
void update( std::vector<float>& layer, int k, int pos, float energy ) {
    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer.size() / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    if ( energy_k >= THRESHOLD / layer.size() || energy_k <= -THRESHOLD / layer.size() )
        layer[k] = layer[k] + energy_k;
}


/* ANCILLARY FUNCTIONS: These are not called from the code section which is measured, leave untouched */
/* DEBUG function: Prints the layer status */
void debug_print(std::vector<float>& layer, std::vector<int>& positions, std::vector<float>& maximum, int num_storms ) {
    int i,k;
    /* Only print for array size up to 35 (change it for bigger sizes if needed) */
    if ( layer.size() <= 35 ) {
        /* Traverse layer */
        for( k=0; k<layer.size(); k++ ) {
            /* Print the energy value of the current cell */
            printf("%10.4f |", layer[k] );

            /* Compute the number of characters. 
               This number is normalized, the maximum level is depicted with 60 characters */
            int ticks = (int)( 60 * layer[k] / maximum[num_storms-1] );

            /* Print all characters except the last one */
            for (i=0; i<ticks-1; i++ ) printf("o");

            /* If the cell is a local maximum print a special trailing character */
            if ( k>0 && k<layer.size()-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
                printf("x");
            else
                printf("o");

            /* If the cell is the maximum of any storm, print the storm mark */
            for (i=0; i<num_storms; i++) 
                if ( positions[i] == k ) printf(" M%d", i );

            /* Line feed */
            printf("\n");
        }
    }
}

/*
 * Function: Read data of particle storms from a file
 */
Storm read_storm_file(char *fname ) {
    FILE *fstorm = fopen( fname, "r" );
    if ( fstorm == NULL ) {
        fprintf(stderr,"Error: Opening storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    Storm storm;    
    int ok = fscanf(fstorm, "%d", &(storm.size) );
    if ( ok != 1 ) {
        fprintf(stderr,"Error: Reading size of storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    storm.posval = new int[storm.size * 2 ];
    if ( storm.posval == NULL ) {
        fprintf(stderr,"Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size );
        exit( EXIT_FAILURE );
    }
    
    int elem;
    for ( elem=0; elem<storm.size; elem++ ) {
        ok = fscanf(fstorm, "%d %d\n", 
                    &(storm.posval[elem*2]),
                    &(storm.posval[elem*2+1]) );
        if ( ok != 2 ) {
            fprintf(stderr,"Error: Reading element %d in storm file %s\n", elem, fname );
            exit( EXIT_FAILURE );
        }
    }
    fclose( fstorm );

    return storm;
}

// Energy relaxation between storms (moving average filter over windowSize elements)
void energy_relaxation(std::vector<float>& layer, const int& windowSize){
        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        std::vector<float> layer_copy = layer; //TODO check if avoiding this malloc speeds things up

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        for(int k=1; k<layer.size()-1; k++ )
            layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;

}

void find_local_maximum(std::vector<float>& layer, float& maximum, int& position ){
    for(int k=1; k<layer.size()-1; k++ ) {
        /* Check it only if it is a local maximum */
        if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
            if ( layer[k] > maximum ) {
                maximum = layer[k];
                position = k;
            }
        }
    }
}
}; //end of namespace MPI_FUNCTIONS