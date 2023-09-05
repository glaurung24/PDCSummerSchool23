#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <cuda.h>
#include "energy_storms_cuda.hpp"

namespace CUDA{

/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

void read_storm_files(int argc,
                    char* argv[], 
                    Storm* storms, 
                    const int& num_storms){
    for(int i=2; i<argc; i++ ) 
        storms[i-2] = read_storm_file( argv[i] );
}




// TODO code from https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu
/**
 * @brief Compute the maximum of 2 single-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
__device__ void atomicMax(float* const address, const float value)
{
    if (*address >= value)
    {
        return;
    }
  
    int* const addressAsInt = (int*)address;
    int old = *addressAsInt, assumed;
  
    do
    {
        assumed = old;
        if (__int_as_float(assumed) >= value)
        {
            break;
        }
  
        old = atomicCAS(addressAsInt, assumed, __float_as_int(value));
    } while (assumed != old);
}

// Kernel does not work with thrust vector (not even if passing raw array)
__global__ void find_local_maximum(
    // const thrust::device_vector<float>& array,
    const float array[],
    const int size,
    float* maximum,
    int* index
    ){
    __shared__ float local_maximum;
    //Add two halo cells to local array (localIdx is maximum THREAD_BLOCK_SIZE)
    __shared__ float local_array[THREAD_BLOCK_SIZE+3];
    int localIdx = threadIdx.x;
    int globalIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if(globalIdx >= size ){
        return;
    }
    local_array[localIdx+1] = array[globalIdx];
    if(localIdx == 0){
        local_maximum = 0.0f;
        if(globalIdx == 0){
            return; //Prevents testing of first site
        }else{
            local_array[0] = array[globalIdx-1];
        }
        
    }
    if(localIdx == THREAD_BLOCK_SIZE){
        local_array[localIdx+2] = array[globalIdx+1];
    }

    __syncthreads();
    if(local_array[localIdx+1] > local_array[localIdx] &&
         local_array[localIdx+1] > local_array[localIdx+2]
         //Prevents the last element in layer to be eligible for local maximum
        && globalIdx < (size -1)){
        if(local_array[localIdx+1] > local_maximum){
            local_maximum = local_array[localIdx+1];
            atomicMax(maximum, local_maximum);
            if(local_maximum == *maximum){
                atomicExch(index, globalIdx);
            }
        }
    }
}

void run_calculation(float* layer, const int& layer_size, Storm* storms, const int& num_storms,
                float* maximum,
                int* positions){
    //lookup table for prefactor 1/sqrt(distance)/layer_size
    //It needs to be 2*layer_size to fit 
    thrust::host_vector<float> look_up;
    look_up.reserve(2*layer_size);
    for(int i = 0; i<2*layer_size; i++){
        look_up.push_back(1.0f/sqrtf((float)abs(layer_size-i)+1)/(float)layer_size); //TODO this should be done on the gpu
    }
    thrust::device_vector<float> look_up_device = look_up;
    thrust::device_vector<float> layer_device(layer_size,0);
    // thrust::device_vector<float> energy_vector_device(layer_size,0);
    // thrust::device_vector<bool> stencil(layer_size);

    /* 4. Storms simulation */
    for(int i=0; i<num_storms; i++) {

        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */

        //TODO parallize this loop more with cuda streams
        for(int j=0; j<storms[i].size; j++ ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position */
            int position = storms[i].posval[j*2]; //TODO check if position is outside of range
            int translated_position = layer_size-position; //relative position to find right part of lookup
            if(translated_position+layer_size > 2*layer_size){
                std::cerr << "position outside of layer" << std::endl;
                exit(EXIT_FAILURE);
            }
            float sign = 1.0f;
            if(energy < 0){
                energy = abs(energy);
                sign = -1.0;
            }

            //Update
            thrust::transform(thrust::device,
                                look_up_device.begin()+translated_position, 
                                look_up_device.begin()+translated_position+layer_size,
                                layer_device.begin(),
                                layer_device.begin(),
                                thrust::placeholders::_2 + 
                                sign*thrust::placeholders::_1*energy*
                                (thrust::placeholders::_1*energy > THRESHOLD) //==0 if energy[k] is below threshold
                            );
        }
        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        thrust::device_vector<float> layer_copy = layer_device;

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        // for(int k=1; k<layer_size-1; k++ )
        //     layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;
        thrust::transform(thrust::device,
                            layer_copy.begin(), 
                            layer_copy.end()-2, 
                            layer_device.begin()+1, 
                            layer_device.begin()+1,
                            thrust::plus<float>()
                        );
        thrust::transform(thrust::device,
                        layer_copy.begin()+2, 
                        layer_copy.end(), 
                        layer_device.begin()+1, 
                        layer_device.begin()+1,
                        (thrust::placeholders::_1 +
                        thrust::placeholders::_2)/3.0f
                    );

        //Find the maximum in layer and its position
        float* maximum_device;
        int* position_device;
        cudaMalloc((void**)&maximum_device, sizeof(float));
        cudaMalloc((void**)&position_device, sizeof(int));

        find_local_maximum<<<ceil(layer_size/(float)THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE>>>(thrust::raw_pointer_cast(layer_device.data()), layer_device.size(), maximum_device, position_device);
        cudaMemcpy(maximum+i, maximum_device, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(positions+i, position_device, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree( maximum_device);
        cudaFree(position_device);
        
    }
    cudaMemcpy(layer, layer_device.data().get(), layer_size*sizeof(float), cudaMemcpyDeviceToHost);
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


}; //end of namespace SEQUENTIAL