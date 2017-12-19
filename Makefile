# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= #-O3 
ARCH   	:= -march=native

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := -fopenmp -lpthread -mavx -lMOAB

# Names of executables to create
EXEC := random_walk seq_tally

# Includes
Linked_Libs := ~/opt/moab/include

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : $(EXEC) test_execs

random_walk : random_walk.cpp
	@ echo Building $@...
	@ $(CXX) $(CXXSTD) -o $@ $< $(LDFLAGS) $(OPT)

seq_tally : seq_tally.cpp
	@ echo Building $@...
	@ $(CXX) $(CXXSTD) -g -I$(Linked_Libs) -L$(Linked_Libs) -o $@ $< $(LDFLAGS) $(OPT)

mesh_ex : StructuredMeshSimple.cpp
	@ echo Building $@...
	@ $(CXX) $(CXXSTD) -I$(Linked_Libs) -L$(Linked_Libs) -o $@ $< $(LDFLAGS) $(OPT) 

convert_h5m :
	$(MOAB_PATH)/bin/mbconvert $(FILE) $(NEWFILE)

# TODO: add targets for building executables

.PHONY: clean
clean:
	@ rm -f $(EXEC) $(OBJS) *.out *.vtk *h5m
