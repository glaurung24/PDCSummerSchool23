#pragma once

#include <vector>

#define THRESHOLD    0.001f
#define MPI_ROOT_PROCESS 0


namespace MPI_FUNCTIONS{
/* Structure used to store data for one storm of particles */
typedef struct {
    int size;    // Number of particles
    int *posval; // Positions and values
} Storm;

typedef struct {
    int size;
    int rank;
    int root = MPI_ROOT_PROCESS;
} MPIInfo;

double cp_Wtime();
void read_storm_files(int argc, 
                    char* argv[], 
                    std::vector<Storm>& storms
                    );
void run_calculation(std::vector<float>& layer, 
                std::vector<Storm>& storms,
                std::vector<float>& maximum,
                std::vector<int>& positions,
                MPIInfo& mpi_info);
void update( std::vector<float>& layer, int k, int pos, float energy );
void debug_print(std::vector<float>& layer,  std::vector<int>& positions,  std::vector<float>& maximum, int num_storms );
Storm read_storm_file( char *fname );
void debug_print(std::vector<float>& layer, int *positions, float *maximum, int num_storms );
Storm read_storm_file(char *fname );
void energy_relaxation(std::vector<float>& layer);
void find_local_maximum(std::vector<float>& layer, float& maximum, int& position );

}; //end of namespace SEQUENTIAL
