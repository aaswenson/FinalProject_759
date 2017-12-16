#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>

class twoDmesh {
    public:
        float* x;
        float* y;
        float* z;
        float* tl;
};

twoDmesh gen_mesh(int NI, int NJ, int NK,
                  float dx, float dy, float dz){
    // create mesh object and allocate memory
    twoDmesh mesh;
    mesh.x = (float*) malloc((NI+1)*sizeof(float));
    mesh.y = (float*) malloc((NJ+1)*sizeof(float));
    mesh.z = (float*) malloc((NK+1)*sizeof(float));
    // assign vertex data to mesh object
    for (unsigned i=0; i<NI+1; i++){mesh.x[i] = i*dx;}
    for (unsigned j=0; j<NJ+1; j++){mesh.y[j] = j*dy;}
    for (unsigned k=0; k<NK+1; k++){mesh.z[k] = j*dz;}

    return mesh;
}

std::pair<int> get_voxel(mesh, position){
    


}

int main(int argc, char* argv[])
{
    if (argc != 5){
        std::cout << "Usage: Nx Ny Nz hx hy hz" << std::endl;
        return 1;
    }
    const unsigned NI = atof(argv[1]);
    const unsigned NJ = atof(argv[2]);
    const unsigned NK = atof(argv[3]);
    const float DX = atof(argv[4]); 
    const float DY = atof(argv[5]); 
    const float DZ = atof(argv[6]);
    
    twoDmesh mesh = gen_mesh(NI, NJ, NZ, DX, DY, );

    for (int i=0; i<NI+1; i++){
        std::cout<< mesh.x[i] << std::endl;
    }

    return 0;
}
