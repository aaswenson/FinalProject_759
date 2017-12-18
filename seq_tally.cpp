#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>

// particle track data
class particleTrack{
    public:
        unsigned int Nx;
        unsigned int Ny;
        unsigned int Nz;
        double* x_pos;
        double* y_pos;
        double* z_pos;
        double* u;
        double* v;
        double* w;
        double* track_length;
        double* energy;
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
    dataTrack.x_pos = (double*) malloc(len * sizeof(double));
    dataTrack.y_pos = (double*) malloc(len * sizeof(double));
    dataTrack.z_pos = (double*) malloc(len * sizeof(double));
    dataTrack.u = (double*) malloc(len * sizeof(double));
    dataTrack.v = (double*) malloc(len * sizeof(double));
    dataTrack.w = (double*) malloc(len * sizeof(double));
    dataTrack.track_length = (double*) malloc(len * sizeof(double));
    dataTrack.energy = (double*) malloc(len * sizeof(double));

    // file instance
    std::fstream fp;

    // open file
    fp.open(filename);


    // read columns from file 
    unsigned i = 0;
    while (!fp.eof()) {
      fp >> dataTrack.x_pos[i] >> dataTrack.y_pos[i] >> dataTrack.z_pos[i] >> dataTrack.u[i] >> dataTrack.v[i] >> dataTrack.w[i] >> dataTrack.track_length[i] >> dataTrack.energy[i];
      i++;
    }

    // close file
    fp.close();

    return dataTrack;
}



int main(int argc, char** argv){

    particleTrack dataTrack = read_array("event_history.txt");
    std::cout << dataTrack.x_pos[3] << " ";

    return 0;
}

