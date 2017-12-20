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

class collision_event{
    
    public:
        float V;
        float x, y, z;
        float checkx, checky, checkz;
        float u, v, w;
        float s;
        int inc_vox[3];
        int vox_ID[3];
        unsigned int i, j, k;

        // place to store voxel surface data
        float x_surfs[2];
        float y_surfs[2];
        float z_surfs[2];

        // remaining track length
        float rtl;
        
        void calc_vox_vol(twoDmesh mesh){V = mesh.dx*mesh.dy*mesh.dz;}

        void start_track(unsigned int trackID, 
                            particleTrack data, twoDmesh mesh){
            
            x = data.x_pos[trackID];
            y = data.y_pos[trackID];
            z = data.z_pos[trackID];
            // load data for new particle (launched from 0,0,0)
            if ((x*x + y*y + z*z) == 0){
                vox_ID[0] = (mesh.NI-1)/2;
                vox_ID[1] = (mesh.NJ-1)/2;
                vox_ID[2] = (mesh.NK-1)/2;
            }

            u = data.u[trackID];
            v = data.v[trackID];
            w = data.w[trackID];
            rtl = data.track_length[trackID];
        }


        void get_voxel_surfs(twoDmesh mesh){
            x_surfs[0] = mesh.x[vox_ID[0]]; x_surfs[1] = mesh.x[vox_ID[0]+1];
            y_surfs[0] = mesh.y[vox_ID[1]]; y_surfs[1] = mesh.y[vox_ID[1]+1];
            z_surfs[0] = mesh.z[vox_ID[2]]; z_surfs[1] = mesh.z[vox_ID[2]+1];
        }

        void eliminate_surfs(){
            
            // based on particle direction, choose three eligible surfaces to
            // check for crossing
            checkx = x_surfs[(int)(u+1)];
            checky = y_surfs[(int)(v+1)];
            checkz = z_surfs[(int)(w+1)];
        }

        void distance_to_cross(){
            // get distance to crossing for each of the three eligible surfaces
            // create the transformation vector to increment voxel_ID
            
            inc_vox[0] = 0; inc_vox[1] = 0; inc_vox[2] = 0;
            
            float sx = (checkx-x)/u;
            float sy = (checky-y)/v;
            float sz = (checkz-z)/w;
            s = sx;
            int inc_idx = 0;
            float inc_val = u;
            
            if (sy < s){
                s = sy; inc_val = v; inc_idx = 1;
            }else if (sz < s){ 
                s = sz; inc_val = w; inc_idx = 2;
            }
            
            if (rtl > s){
                // only increment voxel if we leave the box
                inc_vox[inc_idx] = (int)(inc_val / fabs(inc_val));
            }
        }
        
        void update_tl(twoDmesh mesh){
            if (rtl > s){
                mesh.flux[vox_ID[0] + 
                      vox_ID[1]*mesh.NI + 
                      vox_ID[2]*mesh.NI*mesh.NJ] = s / V;
                // update remaining track length and particle position
                update_pos(s);
                rtl -= s;
            }
            else{
                // expend remaining track length inside voxel
                mesh.flux[vox_ID[0] + 
                      vox_ID[1]*mesh.NI + 
                      vox_ID[2]*mesh.NI*mesh.NJ] = rtl / V;
                // update position
                update_pos(rtl);
                rtl = 0;
            }
        }

        void update_voxel_ID(){
            // increment the voxel ID based on the surface crossed upon voxel
            // exit
            vox_ID[0] += inc_vox[0];
            vox_ID[1] += inc_vox[1];
            vox_ID[2] += inc_vox[2];
        }

        void update_pos(float travel){
            // update the x,y,z position
            x += u*travel;
            y += v*travel;
            z += w*travel;
        }

        void walk_particle(twoDmesh mesh){
            int test = 0;
            while (rtl > 0){
               get_voxel_surfs(mesh);
               eliminate_surfs();
               distance_to_cross();
               update_tl(mesh);
               update_voxel_ID();
               test++;
            }
        }
};


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
    
    return 0;
}
