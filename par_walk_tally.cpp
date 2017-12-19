#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <gen_mesh.cpp>
#include <random_walk.cpp>
#include <read_methods.cpp>

void seq_tally(int N, particleTrack col_data, twoDmesh mesh,
                   int NI, int NJ, int NK){

        collision_event particle;
        particle.calc_vox_vol(mesh);
                
        for (int partID = 0; partID <col_data.Ntracks; partID++){
            particle.start_track(partID, col_data, mesh);
            particle.walk_particle(mesh);
        }       
}


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
    // start tallying
    seq_tally(N, dataTrack, mesh, NI, NJ, NK);
    std::cout << (int) 2/3 << std::endl;
    return 0;
}
