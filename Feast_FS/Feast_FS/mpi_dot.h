#pragma once
#include <mpi.h>
#include <complex>
using namespace std;
void mpi_dot(complex<double>*, complex<double>*, complex<double>*, int, int, int, MPI_Comm);
void mpi_plus(complex<double>*, complex<double>*, complex<double>*, int, int, int, MPI_Comm, bool);
void mpi_times(complex<double>*, complex<double>*, complex<double>, int, int, int, MPI_Comm);