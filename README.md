EP759 Final Project
===================

How to compile the code 
-------------------
In order to compile the code, run make. An executable called par\_tally will be created. 
This executable needs to be run with three arguments:
1. Number of particle 
2. Number of mesh voxels in any direction
3. Voxel size on any direction 

Example: ./par\_tally 50 21 0.5

This excutable will run 100 particles, and create a 50 x 50 x 50 mesh with voxel size 0.5. 

Something to note is that the number of mesh voxels, second input, must always be odd. 


Problem statement 
-----------------
We will investigate various configurations of GPU acceleration for parallelized particle tallying in Monte Carlo physics problems. The goal of this project is to explore computational options and attempt to improve the calculation time in mesh tally problems. 

Motivation/Rationale
--------------------
Mesh tallies from Monte Carlo transport calculations log particles on a geometric mesh in problem space. These tallies are used in conjunction with response functions to determine important reaction rates. These reaction rates are used to draw conclusions and perform analysis on nuclear systems.

Our motivation for this work is to minimize computation time in mesh tally problems. This problem is related to the research fields of the members of this team. We perform Monte Carlo radiation transport calculations for complex nuclear systems. Many of these calculations are slow, hence accelerations on them will help improve performance time.

How we plan to go about it
--------------------------
This project will be executed in several phases. The general scheme of our project is laid out below:

| Tasks                                                                     | Leaders               |
| ------------------------------------------------------------------------- | --------------------- |
| Execute random particle walk in a sample problem space, store the results | Nancy                 |
| Perform mesh tally routine using serial code                              | Alex                  |
| Perform mesh tally routine using CUDA GPU code                            | Alex, YoungHui        |
| Perform reaction rate routine using serial code                           | Nancy                 |
| Perform reaction rate routine using CUDA GPU code                         | Nancy, Alex, YoungHui |
| Execute routines in various acceleration configurations                   | YoungHui              |
| Conduct scaling analysis and report the results                           | Nancy, Alex, YoungHui |

This problem is particularly suitable to GPU parallelization because mesh tally calculations involve identical operations performed repeatedly throughout a mesh. We plan to use GPU parallelization to calculate the mesh tally for every voxel in our problem space, with tally calculations for each voxel occurring on a single GPU thread. Following our mesh tally calculation, we will utilize GPU parallelization to perform various numbers (per voxel) of reaction rate operations on the mesh values. We will parallelize across voxels, across reaction rates, and in some combination of the two to investigate the performance of various acceleration schemes.


How we will demonstrate what you accomplished
---------------------------------------------
The performance of the algorithm (in serial and for each accelerated configuration) is dependent on two main problem parameters: (1) the mesh resolution, (2) the particle mean free path of interaction. These parameters will play an important role in analyzing performance of our code. In addition to these parameters, the performance will be dependent on the success of our acceleration efforts. We will attempt to analyze the effects of both problem parameters and acceleration schemes.

We will demonstrate the success of our GPU acceleration efforts by comparing calculation times to understand how the performance depends on mesh resolution, mean free path, number of mesh voxels across various hardware acceleration configurations (including serial for a baseline).


Team members
------------
We are a team of graduate students in the Department of Engineering Physics. We do research with Professor Paul Wilson in the Computational Nuclear Engineering Research Group (CNERG).

* Alex Swenson: Master’s student in Nuclear Engineering and Engineering Physics.
* Nancy Granda-Duarte: Ph.D student in Nuclear Engineering and Engineering Physics. 
* YoungHui Park: Ph.D student in Nuclear Engineering and Engineering Physics.


Deliverables
------------
- Serial version of code for mesh tallying routine
- GPU-accelerated version of code for mesh tallying routine
- GPU-accelerated version of code for reaction rate routine 
- Final report write-up with results of scaling analysis 

Participate in Rescale sponsored Final Project competition: Yes

Link to your Final Project Repo: https://github.com/aaswenson/FinalProject_759


