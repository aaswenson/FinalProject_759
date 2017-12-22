#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

struct twoDmesh {
        unsigned int N;
        float h;
// lower bounding surface position (in i,j,k) for all voxels
        float* x;
        float* y;
        float* z;
        float* flux;
};

twoDmesh gen_mesh(int N, float h){
    // create mesh object and allocate memory
    twoDmesh mesh;
    mesh.h = h; mesh.N = N;
    mesh.x = (float*) malloc((N+1)*sizeof(float));
    mesh.y = (float*) malloc((N+1)*sizeof(float));
    mesh.z = (float*) malloc((N+1)*sizeof(float));
    mesh.flux = (float*) malloc((N*N*N)*sizeof(float));
    // assign vertex data to mesh object
    for (unsigned i=0; i<N+1; i++){mesh.x[i] = (i - N/2.0)*h;}
    for (unsigned j=0; j<N+1; j++){mesh.y[j] = (j - N/2.0)*h;}
    for (unsigned k=0; k<N+1; k++){mesh.z[k] = (k - N/2.0)*h;}
    // set all flux to 0
    for (unsigned i=0; i<N*N*N;i++){mesh.flux[i] = 0;}
    return mesh;
}
