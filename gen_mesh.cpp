#include "moab/Core.hpp"
// structured mesh interface
#include "moab/ScdInterface.hpp"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>

using namespace moab;

int main(int argc, char* argv[])
{
    if (argc != 7){
        std::cout << "Usage: Nx Ny Nz hx hy hz" << std::endl;
        return 1;
    }

    std::string N = (1, argv[1]);
    std::string h = (1, argv[2]);
    std::string mesh_name = "structured_mesh_" + N + "_" + h + ".vtk";
    const unsigned NI = atoi(argv[1]);
    const unsigned NJ = atoi(argv[2]);
    const unsigned NK = atoi(argv[3]);
    const double DX = atoi(argv[4]); 
    const double DY = atoi(argv[5]);
    const double DZ = atoi(argv[6]);
    
    moab::Core mbcore;
    moab::Interface& mbint = mbcore;

    // moab::ScdInterface, structured mesh interface for MOAB
    moab::ScdInterface *scdint;

    // Pass structured mesh to MOAB for Error Queries
    moab::ErrorCode rval = mbint.query_interface(scdint);
    MB_CHK_SET_ERR(rval, "mbint.query_interface");
    
    // Create the mesh:
    moab::ScdBox *scdbox = NULL;
    rval = scdint->construct_box(moab::HomCoord(0,0,0), 
                     moab::HomCoord(NI,NJ,NK),
                     NULL, 
                     0, 
                     scdbox);
    MB_CHK_SET_ERR(rval, "scdint->construct_box");

    // set the vertices
    for(unsigned i = 0; i < NI+1; i++) 
      for(unsigned j = 0; j < NJ+1; j++)
          for(unsigned k = 0; k < NK+1; k++) {
        // First, get the entity handle:
        moab::EntityHandle handle = scdbox->get_vertex(i,j,k);

        // Compute the coordinate:
        double coord[3] = {DX*i, DY*j, DZ*k};

        // Change the coordinate of the vertex:
        mbint.set_coords(&handle, 1, coord);
      }

    // Attach Tags
    // MB_TAG_DENSE allocates a tag to every mesh point
    // Could be useful going forward to use MB_TAG_SPARSE, which only allocates
    // tag memory as needed.
    moab::Tag temp_tag;
    double temp_default_value = 0.0;
    rval = mbint.tag_get_handle("track length", 1, MB_TYPE_DOUBLE, temp_tag, 
                                MB_TAG_DENSE | moab::MB_TAG_CREAT, 
                        &temp_default_value);
    MB_CHK_SET_ERR(rval, "mbint.tag_get_handle(track length)");
      
    // Write the mesh object to file
    const char* savefile = mesh_name.c_str();
    rval = mbint.write_file(savefile);
    MB_CHK_SET_ERR(rval, "write_file(struct_mesh.vtk)");


    return 0;
}
