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

