# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3 
ARCH   	:= -march=native

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := -fopenmp -lpthread -mavx

# Names of executables to create
EXEC := random_walk

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


# TODO: add targets for building executables

.PHONY: clean
clean:
	@ rm -f $(EXEC) $(OBJS) *.out
