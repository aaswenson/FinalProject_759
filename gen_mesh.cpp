#include "moab/Core.hpp"
// structured mesh interface
#include "moab/ScdInterface.hpp"

#include <iostream>
#include <cmath>

using namespace moab;

int main(int argc, char* argv[])
{
  moab::Core mbcore;
  moab::Interface& mbint = mbcore;

  // ***********************
  // *   Create the Mesh   *
  // ***********************

  // First, lets make the mesh. It will be a 100 by 100 uniform grid
  // (there will be 100x100 quads, 101x101 vertexes) with dx = dy =
  // 0.1. Unlike the previous example, we will first make the mesh,
  // then set the coordinates one at a time.

  const unsigned NI = atoi(argv[1]);
  const unsigned NJ = atoi(argv[2]);
  const unsigned NK = atoi(argv[3]);

  // moab::ScdInterface, structured mesh interface for MOAB
  moab::ScdInterface *scdint;

  // Tell MOAB that our mesh is structured:
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


  // ******************************
  // *   Set Vertex Coordinates   *
  // ******************************
  
  const double DX = 0.1;
  const double DY = 0.1;
  const double DZ = 0.1;

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


  // *******************
  // *   Attach Tags   *
  // *******************
  
  // Create the tags:
  //
  // MB_TAG_DENSE allocates a tag to every mesh point
  // Could be useful going forward to use MB_TAG_SPARSE, which only allocates
  // tag memory as needed.
  moab::Tag temp_tag;
  double temp_default_value = 0.0;
  rval = mbint.tag_get_handle("temperature", 1, MB_TYPE_DOUBLE, temp_tag, 
                              MB_TAG_DENSE | moab::MB_TAG_CREAT, 
			      &temp_default_value);
  MB_CHK_SET_ERR(rval, "mbint.tag_get_handle(temperature)");



  // ***************************
  // *   Write Mesh to Files   *
  // ***************************

  // NOTE: Some visualization software (such as VisIt) may not
  // interpret the velocity tag as a vector and you may not be able to
  // plot it. But you should be able to plot the temperature on top of
  // the mesh.

  rval = mbint.write_file("moabuse4.vtk");
  MB_CHK_SET_ERR(rval, "write_file(moabuse4.vtk)");


  return 0;
}
