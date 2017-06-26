#include <iostream>
#include <time.h>
#include <mpi.h>
#include <cmath>
#include <complex>
#include "mpi_dot.h"
#include "CubeFD.h"
#include "SolveZ.h"
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


	// ============================
	// == Main Execution Section ==
	// ============================

	const int d = 20;
	const int N = d*d*d;
	const int m = 6;
	const int L = 800;
	CubeFD cl(d, d, d, rank, size, MPI_COMM_WORLD);

	// 2d array pointers
	complex<double>** V;
	complex<double>** Z;
	complex<double>** H_ptrs;
	complex<double> H[m + 1][m];
	complex<double> y[m][1];

	// 1d array pointers
	complex<double>* y_ptrs = &y[0][0];
	complex<double>* r0;
	complex<double>* x0;
	complex<double>* w;
	complex<double>* b;
	complex<double>* temp1;
	complex<double>* e1;
	complex<double> beta = 0.0f;
	complex<double> dot_sum = 0.0f;
	complex<double> temp_val;
	complex<double> Ze;

	// Allocate Memory
	if (rank == 0)
	{
		V = new complex<double>*[m + 1];
		Z = new complex<double>*[m + 1];
		H_ptrs = new complex<double>*[m + 1];
		for (int i = 0; i < m + 1; i++)
		{
			V[i] = new complex<double>[N]();
			Z[i] = new complex<double>[N]();
			y[i][0] = (complex<double>) 0.0;
			if (rank == 0)
			{
				H_ptrs[i] = H[i];
				for (int j = 0; j < m; j++)
					H[i][j] = (complex<double>) 0.0;
			}
		}

		r0 = new complex<double>[N];
		w = new complex<double>[N];
		b = new complex<double>[N];
		x0 = new complex<double>[N];
		e1 = new complex<double>[m + 1]();
		temp1 = new complex<double>[N];

		for (int i = 0; i < N; i++)
		{
			b[i] = (complex<double>) (i + 1);
			x0[i] = b[i];
		}
	}

	cout << "DEBUG: Execution made it to here.\n";

	SolveZ(Ze, m, beta, H_ptrs, V, Z, *H, w, b, *y, y_ptrs, e1, x0, r0, temp1, L, N, rank, size, MPI_COMM_WORLD, cl);
	// ============================

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	if (rank == 0) cout << "Time Elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
	return 0;
}