#pragma once
#include <complex>
#include "mpi.h"
#include "CubeFD.h"
using namespace std;
void SolveZ(complex<double> Ze, int m, complex<double> beta, complex<double>** H_ptrs,
			complex<double>** V, complex<double>** Z, complex<double>* H, 
			complex<double>* w, complex<double>* b, complex<double>* y, complex<double>* y_ptrs,
			complex<double>* e1, complex<double>* x0, complex<double>* r0, complex<double>* temp1,
			int L, int N, int rank, int size, MPI_Comm comm, CubeFD cl);