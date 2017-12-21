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

    
__global__  void parallel_walk(unsigned int Ngrid, unsigned int N, float* x, float* y, float* z,
                               float h, float* gflux, 
                               float* x_pos, float* y_pos, float* z_pos, 
                               float* u, float* v, float* w, 
                               float* track_length){
    unsigned int x_idx = threadIdx.x; 
    unsigned int y_idx = threadIdx.y;
    unsigned int z_idx = threadIdx.z;;
    unsigned int tl_ID = (x_idx + y_idx*blockDim.x +
            z_idx*blockDim.x*blockDim.y) +
        blockDim.x*blockDim.y*blockDim.z*blockIdx.x;

    float x_0, y_0, z_0, tl;
    float dir_inv[3];
    float tmin, tmax, savet;
    float x_surfs[2], y_surfs[2], z_surfs[2];
    float V = h*h*h; 
    int intersect;
    gflux[tl_ID] = 0;



    if(x_idx < Ngrid || y_idx < Ngrid || z_idx < Ngrid){

        // get voxel surfaces
        x_surfs[0] = x[x_idx]; x_surfs[1] = x[x_idx+1];
        y_surfs[0] = y[y_idx]; y_surfs[1] = y[y_idx+1];
        z_surfs[0] = z[z_idx]; z_surfs[1] = z[z_idx+1];

        for (int PID=0; PID<N; PID++){
            // get particle track length
            tl = track_length[PID];
            // inverted direction to be used in ray-box intersection check
            dir_inv[0] =  1 / u[PID];
            dir_inv[1] =  1 / v[PID];
            dir_inv[2] =  1 / w[PID];

            // default assumption is we cross into box
            x_0 = x_pos[PID]; y_0 = y_pos[PID]; z_0 = z_pos[PID];
            
            
            // x goes first
            float txmin = (x_surfs[0] - x_0) * dir_inv[0];
            float txmax = (x_surfs[1] - x_0) * dir_inv[0];
            // if necessary swap within x
            if (txmax < txmin){
                savet = txmax;
                txmax = txmin;
                txmin = savet;
            }
            // distance to cross in y
            float tymin = (y_surfs[0] - y_0) * dir_inv[1];
            float tymax = (y_surfs[1] - y_0) * dir_inv[1];
            // if necessary swap within y
            if (tymax < tymin){
                savet = tymax;
                tymax = tymin;
                tymin = savet;
            }
            // distance to cross in z
            float tzmin = (z_surfs[0] - z_0) * dir_inv[2];
            float tzmax = (z_surfs[1] - z_0) * dir_inv[2];
            // if necessary swap within z
            if (tzmax < tzmin){
                savet = tzmax;
                tzmax = tzmin;
                tzmin = savet;
            }

            // maximum min t is the distance to box entry
            tmin = fmax(txmin, fmax(tymin, tzmin));
            // minimum max t is the distance to box exit
            tmax = fmin(txmax, fmin(tymax, tzmax));

            // select cases only where particle was in voxel
            if ( tmin < tmax && tmax > 0 && tl > tmin){
                // particle through entire voxel
                if (tl > tmax && tmin > 0){gflux[tl_ID] += (tmax - tmin) / V;}
                // particle starts inside voxel, leaves
                if (tmin < 0 && tl > tmax){gflux[tl_ID] += tmax / V;}
                // particle starts outside voxel, end inside
                if (tmax > tl && tmin > 0){gflux[tl_ID] += (tl - tmin) / V;}
                
                // particle starts inside, ends inside
                if (tmax > tl && tmin < 0){gflux[tl_ID] += tl / V;}
            }
        }
    }
}

void par_tally(twoDmesh hmesh, particleTrack hdata, int N, float h){
    
    particleTrack ddata = AllocatePtracData(hdata);
    twoDmesh dmesh = AllocateMeshData(hmesh);
    CopyDatatoDevice(ddata, hdata, dmesh, hmesh);
    
    // size of flux memory
    int flux_size = N*N*N*sizeof(float);

    int max_dim = 10;
    int grid_dim = (int) N*N*N/1000+1; //N/max_dim + 1;
    if (N < 11){
        grid_dim = 1;
        max_dim = N;
    }
    std::cout << "block dim " << max_dim << std::endl;
    std::cout << "N blocks " << grid_dim << std::endl;
    dim3 dimBlock(max_dim, max_dim, max_dim);
    //dim3 dimGrid(grid_dim, grid_dim, grid_dim);
    dim3 dimGrid(grid_dim, 1, 1);
    

    parallel_walk<<<dimGrid,dimBlock>>> (N, ddata.Ntracks, dmesh.x, dmesh.y, dmesh.z, 
            h, dmesh.flux, ddata.x_pos, ddata.y_pos, ddata.z_pos,
         ddata.u, ddata.v, ddata.w, ddata.track_length);

    cudaMemcpy(hmesh.flux, dmesh.flux, flux_size, 
			cudaMemcpyDeviceToHost);
    

    cudaFree(dmesh.flux);
    cudaFree(dmesh.x);
    cudaFree(dmesh.y);
    cudaFree(dmesh.z);
    cudaFree(ddata.x_pos);
    cudaFree(ddata.y_pos);
    cudaFree(ddata.z_pos);
    cudaFree(ddata.u);
    cudaFree(ddata.v);
    cudaFree(ddata.w);
    cudaFree(ddata.track_length);
    cudaFree(dmesh.flux);


}


int main(int argc, char* argv[]){

    if (argc != 4){
        std::cout << "Usage: N_particles N h" << std::endl;
        return 1;
    }
    const unsigned Np = atoi(argv[1]);
    const unsigned N = atof(argv[2]);
    const float h = atof(argv[3]); 

    if (N%2==0){
        std::cout << "Mesh dimensions must be odd!" << std::endl;
        return 1;
    }
    // generate track histories
    execute_walk(Np);
    // Load particle collision history
    particleTrack hdata = read_array("event_history.txt");
    // generate mesh
    twoDmesh hmesh = gen_mesh(N, h);
    
    par_tally(hmesh, hdata, N, h); 
    
    for(int i=0;i<N*N*N;i++){
        std::cout << hmesh.flux[i] << std::endl;
    }
    
    free(hmesh.flux);
    
    return 0;
}
