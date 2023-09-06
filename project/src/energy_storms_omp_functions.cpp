#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include "energy_storms_omp.hpp"
/* Headers for the OpenMP assignment versions */
#include<omp.h>

/* Use fopen function in local tests. The Tablon online judge software 
   substitutes it by a different function to run in its sandbox */
#ifdef CP_TABLON
#include "cputilstablon.h"
#else
#define    cp_open_file(name) fopen(name,"r")
#endif
namespace OMP_FUNCTIONS {




/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
void update( float *layer, int layer_size, int k, int pos, float energy ) {
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
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[k] = layer[k] + energy_k;
}


/* ANCILLARY FUNCTIONS: These are not called from the code section which is measured, leave untouched */
/* DEBUG function: Prints the layer status */
void debug_print(int layer_size, float *layer, int *positions, float *maximum, int num_storms ) {
    int i,k;
    /* Only print for array size up to 35 (change it for bigger sizes if needed) */
    if ( layer_size <= 35 ) {
        /* Traverse layer */
        for( k=0; k<layer_size; k++ ) {
            /* Print the energy value of the current cell */
            printf("%10.4f |", layer[k] );

            /* Compute the number of characters. 
               This number is normalized, the maximum level is depicted with 60 characters */
            int ticks = (int)( 60 * layer[k] / maximum[num_storms-1] );

            /* Print all characters except the last one */
            for (i=0; i<ticks-1; i++ ) printf("o");

            /* If the cell is a local maximum print a special trailing character */
            if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
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

void read_storm_files(int argc,
                    char* argv[], 
                    Storm* storms, 
                    const int& num_storms){
    for(int i=2; i<argc; i++ ) 
        storms[i-2] = read_storm_file( argv[i] );
}


/*
 * Function: Read data of particle storms from a file
 */
Storm read_storm_file( char *fname ) {
    FILE *fstorm = cp_open_file( fname );
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

    storm.posval = (int *)malloc( sizeof(int) * storm.size * 2 );
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


void run_calculation(float* layer, const int& layer_size, Storm* storms, 
        const int& num_storms, float* maximum, int* positions) {
    int i, j, k;
    float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
    // 4. Storms simulation
    for( i=0; i<num_storms; i++) {

        // 4.1. Add impacts energies to layer cells
        // For each particle
        // make each thread private and use omp reduce add
        #pragma omp parallel for private(j, k) reduction(+: layer[:layer_size])
        for (j = 0; j < storms[i].size; j++) {
            // Get impact energy (expressed in thousandths) 
            float energy = (float)storms[i].posval[j * 2 + 1] * 1000;
            // Get impact position
            int position = storms[i].posval[j * 2];

            // For each cell in the layer
            for (k = 0; k < layer_size; k++) {
                // Update the energy value for the cell
                update(layer, layer_size, k, position, energy);
            }
        }

        // 4.2. Energy relaxation between storms 
        // 4.2.1. Copy values to the ancillary array 
        #pragma omp parallel for
        for( k=0; k<layer_size; k++ ) 
            layer_copy[k] = layer[k];

        // 4.2.2. Update layer using the ancillary values.
        // Skip updating the first and last positions 
        #pragma omp parallel for
        for( k=1; k<layer_size-1; k++ ) {
            layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;
        }

        // 4.3. Locate the maximum value in the layer, and its position 
        // Define private variables for each thread
        float local_maximum = -1.0f;
        int local_position = -1;

        // #pragma omp parallel private(local_maximum, local_position)
        // {
        //     #pragma omp for
            for (k = 1; k < layer_size - 1; k++) {
                // Check it only if it is a local maximum 
                if (layer[k] > layer[k - 1] && layer[k] > layer[k + 1]) {
                    if (layer[k] > local_maximum) {
                        local_maximum = layer[k];
                        local_position = k;
                    }
                }
            }
 
            // #pragma omp critical
            // // // One thread at a time
            // {
                // Update global maximum and position 
                if (local_maximum > maximum[i]) {
                    maximum[i] = local_maximum;
                    positions[i] = local_position;
                }
            // }
        // }
    }
    free(layer_copy);
}

}; //end of namespace OMP_FUNCTIONS
   
