#include <random>

#define SPHERE_RADIUS 100
#define MEAN_FREE_PATH 3
#define SEED 123
#define SET_COL 15

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

    std::default_random_engine generator;
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
	  logfile << std::left << std::setw(SET_COL) << x
		  << std::left << std::setw(SET_COL) << y
		  << std::left << std::setw(SET_COL) << z
		  << std::left << std::setw(SET_COL) << u
		  << std::left << std::setw(SET_COL) << v
		  << std::left << std::setw(SET_COL) << w
		  << std::left << std::setw(SET_COL) << pl
		  << std::left << std::setw(SET_COL) << E << "\n";
        }
};


double sample_path_length(){
    // Sample path length and return distance traveled along the particle
    // direction

    double rn = static_cast<double>( rand() ) / static_cast<double>(RAND_MAX);
    double path_length = -log(1 - rn);

    return path_length*MEAN_FREE_PATH;
}

std::vector<double> sample_direction(){
    // sample the particle direction in cartesian space, return a direction
    // vector, (u, v, w)

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
    // Performs the random walk for each particle
    // This function calls the required methods to sample the random-walk
    // physics for a single particle. For each path traversed, store the
    // starting location, direction, energy, and path-length

    // intialize particle
    event_log collision;
    // set initial energy, distance, direction
    collision.get_energy();
    collision.get_distance();
    collision.get_direction();
    // save initial particle pos, energy, direction, distance
    collision.save_log(logfile);
    // update location for next interaction
    collision.update_location();

    // run collisions until particle leaves sphere
    while (collision.radius() < SPHERE_RADIUS){
        // update info, save and calculate new location
        collision.get_energy();
        collision.get_distance();
        collision.get_direction();
        collision.save_log(logfile);
        collision.update_location();
    }
}

void execute_walk(int N){
    srand(SEED);
    std::ofstream event_history;
    event_history.open("event_history.txt");
    
    for (int i=0; i < N; i++){
        walk_particle(event_history);
    }

    event_history.close();
}
