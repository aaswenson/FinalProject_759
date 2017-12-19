#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <gen_mesh.cpp>
#include <random_walk.cpp>
#include <read_methods.cpp>
#include <cuda_prep.cu>

__global__ void walktally(){}


int main(int argc, char* argv[]){

    if (argc != 8){
        std::cout << "Usage: N_particles Nx Ny Nz hx hy hz" << std::endl;
        return 1;
    }
    const unsigned N = atoi(argv[1]);
    const unsigned NI = atof(argv[2]);
    const unsigned NJ = atof(argv[3]);
    const unsigned NK = atof(argv[4]);
    const float DX = atof(argv[5]); 
    const float DY = atof(argv[6]); 
    const float DZ = atof(argv[7]);
    if (NI%2 == 0 || NJ%2 == 0 || NK%2 == 0){
        std::cout << "Mesh dimensions must be odd!" << std::endl;
        return 1;
    }
    // generate track histories
    execute_walk(N);
    // Load particle collision history
    particleTrack dataTrack = read_array("event_history.txt");
    // generate mesh
    twoDmesh mesh = gen_mesh(NI, NJ, NK, DX, DY, DZ);
    
    particleTrack data = AllocatePtracData(dataTrack);
    twoDmesh dmesh = AllocateMeshData(mesh);
    CopyDatatoDevice(data, dataTrack, dmesh, mesh);
    walktally<<<1,1>>>();


    return 0;
}
