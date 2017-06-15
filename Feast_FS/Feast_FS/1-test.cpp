#include <iostream>
#include <time.h>
#include <mpi.h>
#include <cmath>
#include <complex>
#include "mpi_dot.h"
#include "CubeFD.h"
#include "mkl_types.h"
#define MKL_Complex16 std::complex<double>
#include "mkl.h"
using namespace std;
int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	
	lapack_int info,lda,ldb,nrhs;
	const lapack_int m = 5;
	const lapack_int n = 4;
	complex<double> A[m][n] = 	{{(complex<double>) 27.5121, (complex<double>) 29.2534, (complex<double>) 28.3541, (complex<double>) 26.2573}, 
			{(complex<double>) 831.634, (complex<double>) 33.1422, (complex<double>) 34.0041, (complex<double>) 33.0985},
			{(complex<double>) 0.0,     (complex<double>) 1129.23, (complex<double>) 36.2517, (complex<double>) 36.9973},
			{(complex<double>) 0.0,     (complex<double>) 0.0,     (complex<double>) 1341.22, (complex<double>) 39.1992},
			{(complex<double>) 0.0,     (complex<double>) 0.0,     (complex<double>) 0.0,     (complex<double>) 1571.27}};
	complex<double> b[5][1] = {{(complex<double>) 92700800.0}, {(complex<double>) 0.0}, {(complex<double>) 0.0}, {(complex<double>) 0.0}};
	lda = 4;
	ldb = 1;
	nrhs = 1;

	info = LAPACKE_zgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,*A,lda,*b,ldb);

	cout << "y = [";
	for (int i = 0; i < n; i++)
		cout << b[i][0] << " ";
	cout << "]\n";

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	if (rank == 0) cout << "Time Elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
	return 0;
}