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
LDFLAGS := -fopenmp -lpthread -mavx
BIN = "/usr/local/gcc/6.4.0/bin/gcc"

# Names of executables to create
EXEC := random_walk seq_tally par_walk_tally

# Includes
Linked_Libs := ~/opt/moab/include

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : $(EXEC) test_execs

seq_tally : seq_tally.cpp
	@ echo Building $@...
	@ $(CXX) $(CXXSTD) -g -I$(Linked_Libs) -L$(Linked_Libs) -o $@ $< $(LDFLAGS) $(OPT)

par_walk: par_walk_tally.cu 
	nvcc -o par_walk_tally $(OPT) $(CXXSTD) par_walk_tally.cu

# TODO: add targets for building executables

.PHONY: clean
clean:
	@ rm -f $(EXEC) $(OBJS) *.out event_history.txt
