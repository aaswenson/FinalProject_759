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
/*    // Read event_history.txt and get lines.
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
*/
