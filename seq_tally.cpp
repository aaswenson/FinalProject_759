#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <gen_mesh.cpp>

// particle track data
struct particleTrack{
        float* x_pos;
        float* y_pos;
        float* z_pos;
        float* u;
        float* v;
        float* w;
        float* track_length;
        float* energy;
        unsigned int Ntracks;
};

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

        void start_particle(unsigned int trackID, 
                            particleTrack data, twoDmesh mesh){
            
            // load data for new particle (launched from 0,0,0)
            vox_ID[0] = (mesh.NI-1)/2; 
            vox_ID[1] = (mesh.NJ-1)/2; 
            vox_ID[2] = (mesh.NK-1)/2;
            
            x = data.x_pos[trackID];
            y = data.y_pos[trackID];
            z = data.z_pos[trackID];
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
            int inc_idx = 0;
            float inc_val = u;
            
            s = (checkx-x)/u;
            
            if (s > (checky-y)/v){ 
                s = (checky-y)/v;
                inc_val = v;
                inc_idx = 1;
            }
            if (s > (checkz-z)/w){
                s = (checkz-z)/w;
                inc_val = w;
                inc_idx = 2;
            }
            inc_vox[inc_idx] = inc_val / fabs(inc_val);
        }
        
        void update_tl(twoDmesh mesh){
            if (rtl > s){
                mesh.flux[vox_ID[0] + 
                      vox_ID[1]*mesh.NJ + 
                      vox_ID[2]*mesh.NJ*mesh.NK] = s / V;
                // update remaining track length
                rtl -= s;
            }
            else{
                // expend remaining track length inside voxel
                mesh.flux[vox_ID[0] + 
                      vox_ID[1]*mesh.NJ + 
                      vox_ID[2]*mesh.NJ*mesh.NK] = rtl / V;
            }
        }

        void update_voxel_ID(){
            // increment the voxel ID based on the surface crossed upon voxel
            // exit
            std::transform(&inc_vox[0], &inc_vox[2], 
                           &vox_ID[0], 
                           &vox_ID[0], 
                           std::plus<int>());
        }

        void update_pos(){
            // update the x,y,z position
            x += u*s;
            y += v*s;
            z += w*s;
        }

        
};

// particle track file length 
int file_len(const char* filename){

    // file instance
    std::fstream fp;

    // open file
    fp.open(filename);

    std::string line;
    int len = 0;
    // read the number of lines in file
    while (std::getline(fp, line)){
        ++len;
    }

    return len;
}

particleTrack read_array(const char* filename) {

    // create instance of particle Track 
    particleTrack dataTrack;

    // memory allocation
    int len = file_len(filename);
    dataTrack.x_pos = (float*) malloc(len * sizeof(float));
    dataTrack.y_pos = (float*) malloc(len * sizeof(float));
    dataTrack.z_pos = (float*) malloc(len * sizeof(float));
    dataTrack.u = (float*) malloc(len * sizeof(float));
    dataTrack.v = (float*) malloc(len * sizeof(float));
    dataTrack.w = (float*) malloc(len * sizeof(float));
    dataTrack.track_length = (float*) malloc(len * sizeof(float));
    dataTrack.energy = (float*) malloc(len * sizeof(float));

    // file instance
    std::fstream fp;

    // open file
    fp.open(filename);
    
    // save number of tracks
    dataTrack.Ntracks = len;

    // read columns from file 
    unsigned i = 0;
    while (!fp.eof()) {
      fp >> dataTrack.x_pos[i] >> 
            dataTrack.y_pos[i] >> 
            dataTrack.z_pos[i] >> 
            dataTrack.u[i] >> 
            dataTrack.v[i] >> 
            dataTrack.w[i] >> 
            dataTrack.track_length[i] >> 
            dataTrack.energy[i];
      i++;
    }

    // close file
    fp.close();

    return dataTrack;
}

void seq_tally(particleTrack col_data, twoDmesh mesh,
                   int NI, int NJ, int NK){

        collision_event particle;
        particle.calc_vox_vol(mesh);
        particle.start_particle(0, col_data, mesh); 
        particle.get_voxel_surfs(mesh);
        particle.eliminate_surfs();
        particle.distance_to_cross();
        particle.update_tl(mesh);
        particle.update_voxel_ID();
        particle.update_pos();
}


int main(int argc, char* argv[]){

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
    
    // Load particle collision history
    particleTrack dataTrack = read_array("event_history.txt");
    // generate mesh
    twoDmesh mesh = gen_mesh(NI, NJ, NK, DX, DY, DZ);
    // start tallying
    seq_tally(dataTrack, mesh, NI, NJ, NK);

    return 0;
}
