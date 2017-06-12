#pragma once
#include <mpi.h>
void mpi_dot(double*, double*, double*, int, int, int, MPI_Comm);
void mpi_plus(double*, double*, double*, int, int, int, MPI_Comm, bool);
void mpi_times(double*, double*, double, int, int, int, MPI_Comm);