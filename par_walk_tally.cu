#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <gen_mesh.cpp>
#include <random_walk.cpp>
#include <read_methods.cpp>
#include <cuda_prep.cu>

    
__global__  void parallel_walk(unsigned int N, float* x, float* y, float* z,
        float* gflux, float* x_pos, float* y_pos, float* z_pos, 
        float* us, float* vs, float* ws, float* track_length){
//    extern volatile __shared__ float flux[];

    unsigned int x_idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y_idx = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int z_idx = threadIdx.z + blockIdx.z*blockDim.z;
    unsigned int tl_ID = x_idx + y_idx*N + z_idx*N*N;

    float x_0, y_0, z_0, x_1, y_1, z_1, dx, dy, dz, u, v, w;
    float tmin,tmax, t0y, t1y, t0z, t1z, st1;
    float x_surfs[2], y_surfs[2], z_surfs[2];
    float rtl;
    float flux;

    bool intersect;

    // get lower-bound surfaces
    x_surfs[0] = x[x_idx]; x_surfs[1] = x[x_idx+1];
    y_surfs[0] = y[y_idx]; y_surfs[1] = y[y_idx+1];
    z_surfs[0] = z[z_idx]; z_surfs[1] = z[z_idx+1];
    // get the box dimension
    dx = fabsf(x_surfs[0] - x_surfs[1]);
    dy = fabsf(y_surfs[0] - y_surfs[1]);
    dz = fabsf(z_surfs[0] - z_surfs[1]);

    if(x_idx < N && y_idx < N && z_idx < N){
        

        for (int PID=0; PID<N; PID++){
            // default assumption is we cross into box
            intersect = true;
            x_0 = x[PID]; y_0 = y[PID]; z_0 = z[PID];
            u = us[PID]; v = vs[PID]; w = ws[PID];
            
            // x goes first
            tmin = (x_surfs[0] - x_0) / u;
            tmax = (x_surfs[1] - x_0) / u;
            // swap if necessary
            if(tmax < tmin){
                st1 = tmax;
                tmax = tmin;
                tmin = st1;
            }
            // y-direction
            t0y = (y_surfs[0] - y_0) / v;
            t1y = (y_surfs[1] - y_0) / v;
            // swap if necessary
            if(t1y < t0y){
                st1 = t1y;
                t1y = t0y;
                t0y = st1;
            }
            
            if ( (tmin > t1y) || (t0y > tmax) ) {
                intersect = false;
            }else{
                // update min crossing if necessary
                if (t0y > tmin){tmin = t0y;}
                if (t1y < tmax){tmax = t1y;}
                // z-direction
                t0z = (z_surfs[0] - z_0) / w;
                t1z = (z_surfs[1] - z_0) / w;
                // swap if necessary
                if(t1z < t0z){
                    st1 = t1z;
                    t1z = t0z;
                    t0z = st1;
                }
                if( (tmin > t1z) || (t0z > tmax) ){
                    intersect = false;
                }else{
                    // update min crossing if necessary
                    if (t0z > tmin){tmin = t0z;}
                    if (t1z < tmax){tmax = t1z;}
                    gflux[tl_ID] = (tmax - tmin) / (dx*dy*dz);
                }
            }
            
        } 
        
    }


}


int main(int argc, char* argv[]){

    if (argc != 4){
        std::cout << "Usage: N_particles N h" << std::endl;
        return 1;
    }
    const unsigned N_parts = atoi(argv[1]);
    const unsigned N = atof(argv[2]);
    const float h = atof(argv[3]); 
    // uniform mesh
    const unsigned NI = N;
    const unsigned NJ = N;
    const unsigned NK = N;
    const float DX = h;
    const float DY = h;
    const float DZ = h;
    // size of flux memory
    int flux_size = N*N*N*sizeof(float);
    std::cout << flux_size << std::endl;
    if (NI%2 == 0 || NJ%2 == 0 || NK%2 == 0){
        std::cout << "Mesh dimensions must be odd!" << std::endl;
        return 1;
    }
    // generate track histories
    execute_walk(N);
    // Load particle collision history
    particleTrack ddata = read_array("event_history.txt");
    // generate mesh
    twoDmesh hmesh = gen_mesh(NI, NJ, NK, DX, DY, DZ);
    
    particleTrack hdata = AllocatePtracData(ddata);
    twoDmesh dmesh = AllocateMeshData(hmesh);
    CopyDatatoDevice(ddata, hdata, dmesh, hmesh);
    
    int max_dim = 10;
    int grid_dim = (NI / max_dim) + 1;

    dim3 dimBlock(max_dim, max_dim, max_dim);
    dim3 dimGrid(grid_dim, grid_dim, grid_dim);


    parallel_walk<<<dimGrid,dimBlock>>>
        (N, dmesh.x, dmesh.y, dmesh.z, dmesh.flux,
         ddata.x_pos, ddata.y_pos, ddata.z_pos,
         ddata.u, ddata.v, ddata.w, ddata.track_length);

    
	cudaMemcpy(hmesh.flux, dmesh.flux, flux_size , 
			cudaMemcpyDeviceToHost);
    
    for(int i=0;i<N*N*N;i++){
        std::cout << hmesh.x[i] << std::endl;
    }

    return 0;
}
