#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

class twoDmesh {
    public:
        unsigned int Nx;
        unsigned int Ny;
        unsigned int Nz;
        float x_0;
        float y_0;
        float z_0;
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
    for (unsigned k=0; k<NK+1; k++){mesh.z[k] = k*dz;}

    return mesh;
}

std::vector<int> get_voxel(std::vector<float> position, 
                           float dx, float dy, float dz){
    
    std::vector<int> voxel_ID(3);
    // x position
    if (fmod(position[0], dx) == 0) {voxel_ID[0] = position[0] / dx;}
    else {voxel_ID[0] = fmod(position[0], dx) + 1;}
    // y position
    if (fmod(position[1], dy) == 0) {voxel_ID[1] = position[1] / dy;}
    else {voxel_ID[0] = fmod(position[0], dy) + 1;}
    // z position
    if (fmod(position[2], dz) == 0) {voxel_ID[2] = position[2] / dz;}
    else {voxel_ID[2] = fmod(position[2], dz) + 1;}

    return voxel_ID;
    
}


int main(int argc, char* argv[])
{
    if (argc != 7){
        std::cout << "Usage: Nx Ny Nz hx hy hz" << std::endl;
        return 1;
    }
    const unsigned NI = atof(argv[1]);
    const unsigned NJ = atof(argv[2]);
    const unsigned NK = atof(argv[3]);
    const float DX = atof(argv[4]); 
    const float DY = atof(argv[5]); 
    const float DZ = atof(argv[6]);
    const float X0 = atof(argv[6]);
    const float Y0 = atof(argv[6]);
    const float Z0 = atof(argv[6]);
    
    twoDmesh mesh = gen_mesh(NI, NJ, NK, DX, DY, DZ);

    // Read event_history.txt and get lines.
    lines = read_hist('event_history.txt');

    for line in lines {
	p = (x,y,z);  // Initial poistion
	d = (u,v,w);  // Direction
	l = track_length;  // Total track length in the direction.
	(i,j,k) = get_voxel_indices(mesh, p);  // Get mesh indices of initial point. straightforward-version.
	(xs,ys,zs), ls = calc_crossing(mesh, (i,j,k), p, d, l);  // Get crossing point and dist2surf.
	save_value_in_mesh(mesh, (i,j,k), ls);  // Store track length(dist2surf) in the mesh.

	while ls < l{  // Compare dist2surf with remaining track length.
	    (x,y,z) = (xs,ys,zs);  // Reset starting point
	    l = l - ls // Reduce track length.
	    (i,j,k) = march_mesh(mesh, (xc,yc,zc));  // Increase mesh indices.
	    (xs,ys,zs), ls = calc_crossing(mesh, (i,j,k), p, d, l);  // Get crossing point and dist2surf.
	    save_value_in_mesh(mesh, (i,j,k), ls)  // Store track length(dist2surf) in the mesh.
	  }
      }

    return 0;
}
