#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <random>

#define SPHERE_RADIUS 1000
#define MEAN_FREE_PATH 10
#define SEED 123

std::vector<double> sample_direction();
double sample_path_length();

class event_log {
    // This class stores information about particle collision events. It also
    // updates particle radius and location based on sampled collision data.

    public:
    // store path length
        double pl;
    //  collision location
        double x = 0; double y = 0; double z = 0;
    // particle direction
        double u; double v; double w;
    // store particle energy
        double E;

    // Get particle distance
        void get_distance(){ pl = sample_path_length();}
    // Get particle direction
        void get_direction(){
            std::vector<double> direction = sample_direction();
            u = direction[0];
            v = direction[1];
            w = direction[2];
        }
    // Get particle energy
        void get_energy(){ 
            // set normal dist for E sampling
            std::default_random_engine generator;
            std::normal_distribution<double> normal_dist(5.0,2.0);
            E = normal_dist(generator);
        }
    // calculate particle radius
        double radius() {return sqrt(x*x + y*y + z*z);}
    // update particle location
        void update_location() {
            x += pl*u;
            y += pl*v;
            z += pl*w;
        }

    // save particle log
        void save_log(std::ofstream& logfile){
        logfile << std::left << x << "     " << y << "     " << z << "     ";
        logfile << std::left << u << "     " << v << "     " << w << "     ";
        logfile << std::left << pl << "     ";
        logfile << std::left << E << "\n";
        }
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
    // set initial energy, distance, direction
    collision.get_energy();
    collision.get_distance();
    collision.get_direction();
    // save initial particle pos, energy
    collision.save_log(logfile);
    // update location for next interaction
    collision.update_location();

    while (collision.radius() < SPHERE_RADIUS){

        // update info, save and calculate new location
        collision.get_energy();
        collision.get_distance();
        collision.get_direction();
        collision.save_log(logfile);
        collision.update_location();
    }

}



int main(int argc, char** argv){
    if(argc != 2){
        std::cout << "Usage N_particles" << std::endl;
    }
    int N = atoi(argv[1]);
    int i;
    srand(SEED);
    std::ofstream event_history;
    event_history.open("event_history.txt");
    
    for (i=0; i < N; i++){
        walk_particle(event_history);
    }
    event_history.close();

    return 0;

}
