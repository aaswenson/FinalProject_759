# Warnings
WFLAGS  := -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT             := -O3
ARCH    := -march=native

# Language standard
CCSTD   := -std=c99
CXXSTD  := -std=c++11

# Linker options
LDOPT   := $(OPT)
LDFLAGS := -fopenmp -lpthread -mavx
BIN = "/usr/local/gcc/6.4.0/bin/gcc"

# Names of executables to create
EXEC := par_walk

# Includes
Linked_Libs := ~/opt/moab/include

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : $(EXEC) par_walk_tally


par_walk: par_walk_tally.cu
	module load cuda;nvcc -g -o par_walk_tally $(OPT) $(CXXSTD) par_walk_tally.cu -ccbin $(BIN)

# TODO: add targets for building executables

.PHONY: clean
clean:
@ rm -f $(EXEC) $(OBJS) *.out event_history.txt

