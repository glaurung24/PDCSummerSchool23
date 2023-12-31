#pragma once

// #include <thrust/device_vector.h>

#define THRESHOLD    0.001f
#define AVERAGING_WINDOW_SIZE 3
#define THREAD_BLOCK_SIZE 256


namespace CUDA{
/* Structure used to store data for one storm of particles */
typedef struct {
    int size;    // Number of particles
    int *pos; // Positions and values
    int *val;
} Storm;

double cp_Wtime();
void read_storm_files(int argc, 
                    char* argv[], 
                    Storm* storms, 
                    const int& num_storms
                    );
void run_calculation(float* layer, const int& layer_size, Storm* storms, const int& num_storms,
                float* maximum,
                int* positions);
void update( float *layer, int layer_size, int k, int pos, float energy );
void debug_print(int layer_size, float *layer, int *positions, float *maximum, int num_storms );
Storm read_storm_file( char *fname );
void debug_print(int layer_size, float *layer, int *positions, float *maximum, int num_storms );
Storm read_storm_file(char *fname );
// void energy_relaxation(thrust::device_vector<float>& layer);
// void find_local_maximum(float* layer, const int& layer_size, float& maximum, int& position );

}; //end of namespace CUDA