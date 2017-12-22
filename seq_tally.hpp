class collision_event{
    public:
        twoDmesh mesh;
        collision_event(twoDmesh load_mesh){
            mesh = load_mesh;
            N = mesh.N;
            dvox = mesh.h;
            V = dvox*dvox*dvox;
         }
        unsigned int N;
        float dvox; 
        float V;
        float x, y, z;
        float checkx, checky, checkz;
        float u, v, w;
        float s, sx, sy, sz;
        int inc_vox[3];
        int vox_ID[3];
        unsigned int i, j, k;
        // place to store voxel surface data
        float x_surfs[2];
        float y_surfs[2];
        float z_surfs[2];

        // remaining track length
        float rtl;
        
        void start_track(unsigned int trackID, particleTrack data){
            
            x = data.x_pos[trackID];
            y = data.y_pos[trackID];
            z = data.z_pos[trackID];
            // load data for new particle
            vox_ID[0] = (int)((floor(x/(dvox/2)) + N)/2);
            vox_ID[1] = (int)((floor(y/(dvox/2)) + N)/2);
            vox_ID[2] = (int)((floor(z/(dvox/2)) + N)/2);
            u = data.u[trackID];
            v = data.v[trackID];
            w = data.w[trackID];
            rtl = data.track_length[trackID];
        }


        void get_voxel_surfs(){
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
            
            sx = (checkx-x)/u;
            sy = (checky-y)/v;
            sz = (checkz-z)/w;
            
            s = std::min(sx, std::min(sy, sz));

            if (rtl > s){
                if (sx == s){inc_vox[0] = (u > 0) ? 1:-1;}
                if (sy == s){inc_vox[1] = (v > 0) ? 1:-1;}
                if (sz == s){inc_vox[2] = (w > 0) ? 1:-1;}
            }
        }
        
        void update_tl(){
            int tl_idx = vox_ID[0] + 
                              vox_ID[1]*N + 
                              vox_ID[2]*N*N;
            if (rtl > s){
                mesh.flux[tl_idx] += s / V;

                // update remaining track length and particle position
                update_pos(s);
                rtl -= s;
            
            }else{
                // expend remaining track length inside voxel
                mesh.flux[tl_idx]  += rtl / V; 
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

        void walk_particle(){
            // break if we leave mesh
            while ((rtl > 0) && 
                   (vox_ID[0] < N) && 
                   (vox_ID[1] < N) &&
                   (vox_ID[2] < N) ){

               get_voxel_surfs();
               eliminate_surfs();
               distance_to_cross();
               update_tl();
               update_voxel_ID();
              }
            }
        
};


void seq_tally(int N, particleTrack col_data, twoDmesh mesh){

        collision_event particle(mesh);
                
        for (int partID = 0; partID <col_data.Ntracks; partID++){
            if(abs(col_data.x_pos[partID]) < abs(mesh.x[0]) && 
               abs(col_data.y_pos[partID]) < abs(mesh.y[0]) &&
               abs(col_data.z_pos[partID]) < abs(mesh.z[0])){
               particle.start_track(partID, col_data);
               particle.walk_particle();
            }
        }
}
