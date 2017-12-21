#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

struct twoDmesh {
        unsigned int NI;
        unsigned int NJ;
        unsigned int NK;
        float dx;
        float dy;
        float dz;
// lower bounding surface position (in i,j,k) for all voxels
        float* x;
        float* y;
        float* z;
        float* flux;
};

twoDmesh gen_mesh(int NI, int NJ, int NK,
                  float dx, float dy, float dz){
    // create mesh object and allocate memory
    twoDmesh mesh;
    mesh.dx = dx; mesh.dy = dy; mesh.dz = dz;
    mesh.NI = NI; mesh.NJ = NJ; mesh.NK = NK;
    mesh.x = (float*) malloc((NI+1)*sizeof(float));
    mesh.y = (float*) malloc((NJ+1)*sizeof(float));
    mesh.z = (float*) malloc((NK+1)*sizeof(float));
    mesh.flux = (float*) malloc((NI*NJ*NK)*sizeof(float));
    // assign vertex data to mesh object
    for (unsigned i=0; i<NI+1; i++){mesh.x[i] = (i - NI/2.0)*dx;}
    for (unsigned j=0; j<NJ+1; j++){mesh.y[j] = (j - NJ/2.0)*dy;}
    for (unsigned k=0; k<NK+1; k++){mesh.z[k] = (k - NK/2.0)*dz;}

    return mesh;
}
