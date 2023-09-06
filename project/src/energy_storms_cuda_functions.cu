#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include <vector>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
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

// Kernel does not work with thrust vector (not even if passing raw array)
__global__ void energy_relaxation(
    float* array,
    const int size
    ){
    //Add two halo cells to local array
    //Acts also as a local copy
    __shared__ float local_array[THREAD_BLOCK_SIZE+2];
    int localIdx = threadIdx.x;
    int globalIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if(globalIdx >= size ){
        return;
    }
    local_array[localIdx+1] = array[globalIdx];
    if(localIdx == 0){
        if(globalIdx == 0){
            return; //Prevents update of first site
        }else{
            local_array[0] = array[globalIdx-1];
        }
        
    }
    __syncthreads();
    if(globalIdx >= (size - 1)){ //Don't update last element in layer
        return;
    }
    array[globalIdx] = (local_array[localIdx] +
                        local_array[localIdx+1] +
                        local_array[localIdx + 2])/3;
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

__global__ void find_local_maximum(
    const float array[],
    const int size,
    float* maximum,
    int* index
    ){
    __shared__ float local_maximum;
    //Add two halo cells to local array
    __shared__ float local_array[THREAD_BLOCK_SIZE+2];
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
    __syncthreads();
    if(local_array[localIdx+1] > local_array[localIdx] &&
         local_array[localIdx+1] > local_array[localIdx+2]
         //Prevents the last element in layer to be eligible for local maximum
        && globalIdx < (size -1)){
        //First check if found maximum is larger than the maximum in local memory
        //to avoid unneccesary calls to atomicMax 
        if(local_array[localIdx+1] > local_maximum){
            local_maximum = local_array[localIdx+1];
            atomicMax(maximum, local_maximum);
            if(local_maximum == *maximum){
                atomicExch(index, globalIdx);
            }

        }
    }
}

//Note: size is half of the dimension of array!
__global__ void generate_look_up(
    float array[],
    const int size
    ){
    for(int i = threadIdx.x + blockIdx.x*blockDim.x;
        i<2*size;
        i+=blockDim.x * gridDim.x){
        array[i] = (1.0f/sqrtf((float)abs(size-i)+1.0f)/(float)size);
    }
}

void run_calculation(float* layer, const int& layer_size, Storm* storms, const int& num_storms,
                float* maximum,
                int* positions){

    float* look_up_device;
    float* layer_device;
    //Allocate device memory
    if(cudaMalloc((void**)&look_up_device, 2*sizeof(float)*layer_size)
        != cudaSuccess ){
        std::cerr << "Error during allocation of device memory" << std::endl;
        exit(EXIT_FAILURE);
    }
    if(cudaMalloc((void**)&layer_device, sizeof(float)*layer_size)
        != cudaSuccess ){
        std::cerr << "Error during allocation of device memory" << std::endl;
        cudaFree( look_up_device);
        exit(EXIT_FAILURE);
    }
    //Initialize layer_device
    cudaMemset(layer_device, 0.0f, sizeof(float)*layer_size);
    //generate lookup on device
    //lookup table for prefactor 1/sqrt(distance)/layer_size
    //It needs to be 2*layer_size to fit 
    int nr_blocks = ceil(2*layer_size/(float)THREAD_BLOCK_SIZE);
    generate_look_up<<<nr_blocks, THREAD_BLOCK_SIZE>>>(look_up_device, layer_size);

    /* 4. Storms simulation */
    for(int i=0; i<num_storms; i++) {

        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        for(int j=0; j<storms[i].size; j++ ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position */
            int position = storms[i].posval[j*2];
            int translated_position = layer_size-position; //relative position to find right part of lookup
            if(position >= layer_size){
                std::cerr << "position outside of layer" << std::endl;
                cudaFree(layer_device);
                cudaFree(look_up_device);
                exit(EXIT_FAILURE);
            }
            float sign = 1.0f;
            if(energy < 0){
                energy = abs(energy);
                sign = -1.0;
            }
            //Update
                thrust::transform(thrust::device,
                    look_up_device+translated_position, 
                    look_up_device+translated_position+layer_size,
                    layer_device,
                    layer_device,
                    thrust::placeholders::_2 + 
                    sign*thrust::placeholders::_1*energy*
                    (thrust::placeholders::_1*energy > THRESHOLD) //==0 if energy[k] is below threshold
                );
            }
        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array (now done in cuda __shared__ memory*/

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        
        nr_blocks = ceil(layer_size/(float)THREAD_BLOCK_SIZE);
        energy_relaxation<<<nr_blocks, THREAD_BLOCK_SIZE>>>(layer_device, layer_size);

        //Find the maximum in layer and its position
        float* maximum_device;
        int* position_device;
        cudaMalloc((void**)&maximum_device, sizeof(float));
        cudaMalloc((void**)&position_device, sizeof(int));

        find_local_maximum<<<nr_blocks, THREAD_BLOCK_SIZE>>>(layer_device, layer_size, maximum_device, position_device);
 
        //Copy maximum and positions into corresponding host memory
        cudaMemcpy(maximum+i, maximum_device, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(positions+i, position_device, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree( maximum_device);
        cudaFree(position_device);
        
    }
    cudaMemcpy(layer, layer_device, layer_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(layer_device);
    cudaFree(look_up_device);
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