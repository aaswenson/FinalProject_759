#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>

#define SPHERE_RADIUS 1000
#define MEAN_FREE_PATH 10

class event_log {
    public:
// store path length
    double pl;
//  store collision location
    double x = 0;
    double y = 0;
    double z = 0;
// calculate particle radius
    double radius() {return sqrt(x*x + y*y + z*z);}
// store particle direction
    double u;
    double v;
    double w;
// store particle energy
    double E;
};


double sample_path_length(){
    
    double rn = static_cast<double>( rand() ) / static_cast<double>(RAND_MAX);
    double path_length = -log(1 - rn);

    return path_length*MEAN_FREE_PATH;
}

std::vector<double> sample_direction(){
    // sample the particle direction 
    double psi = static_cast<double>( rand() ) / static_cast<double>(RAND_MAX);
    double eta = static_cast<double>( rand() ) / static_cast<double>(RAND_MAX);
    
    std::vector<double> direction(3);
    double pi = 4*atan(1);
    double w = 2*psi - 1;
    double u = sqrt(1-w*w)*cos(2*pi*eta);
    double v = sqrt(1-w*w)*sin(2*pi*eta);

    direction[0] = u;
    direction[1] = v;
    direction[2] = w;

    return direction;
}

void walk_particle(std::ofstream& logfile){
    // intialize particle
    event_log collision;
    
    while (collision.radius() < SPHERE_RADIUS){
        // sample path length
        collision.pl = sample_path_length();

        // Get particle direction
        std::vector<double> direction = sample_direction();
        collision.u = direction[0];
        collision.v = direction[1];
        collision.w = direction[2];

        // Get new particle location
        collision.x += collision.u * collision.pl;
        collision.y += collision.v * collision.pl;
        collision.z += collision.w * collision.pl;

        // write the event history to the log file
        logfile << std::left << collision.x << "     " << 
                   collision.y << "     " << 
                   collision.z << "     ";
        logfile << std::left << collision.u << "     " << 
                   collision.v << "     " << 
                   collision.w << "     \n";
    }

}



int main(int argc, char** argv){
    if(argc != 2){
        std::cout << "Usage N_particles" << std::endl;
    }
    int N = atoi(argv[1]);
    int i;

    std::ofstream event_history;
    event_history.open("event_history.txt", std::ofstream::app);
    
    for (i=0; i < N; i++){
        walk_particle(event_history);
    }
    event_history.close();

    return 0;

}
