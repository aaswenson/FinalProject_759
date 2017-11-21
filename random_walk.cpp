#include <iostream>
#include <cstdio>
#include <cmath>

using namespace std;



double sample_path_length(){
    
    double rn = static_cast<double>( rand() ) / static_cast<double>(RAND_MAX);
    double path_length = math::log(1 - rn);

    return path_length
}

vector<double> sample_direction(){

}





int main(int argc, char* argv){
    if(argc != 2){
        cout << "Usage N = number of particles" << endl;
    }

    int N = atoi(argv[1]);
    double path_length;

    for (i=0; i < N, i++){
        path_length = sample_path_length();
        cout << path_length << endl;
    }

    return 0;

}
