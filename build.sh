mpicxx -o run mpi_dot.cpp CubeFD.cpp 4-MF_GMRES.cpp -L$MKLROOT/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -Wl,-rpath=$MKLROOT/lib/intel64 -std=c++11 -fopenmp
