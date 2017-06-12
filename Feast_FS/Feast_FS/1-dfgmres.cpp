#include <iostream>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include <algorithm>
#include "mpi_dot.h"
#include "CubeFD.h"
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

	// === Main Execution Section ===
	const int d = 3;
	const MKL_INT N = d*d*d;
	double* X;
	double* B;
	MKL_INT RCI_request;
	MKL_INT* ipar;
	double* dpar;
	double* tmp;
	MKL_INT itercount = 0;
	CubeFD cl(d, d, d, rank, size, MPI_COMM_WORLD);

	if (rank == 0)
	{
		X = new double[N];
		B = new double[N];
		ipar = new MKL_INT[128];
		dpar = new double[128];
		tmp = new double[((2 * min(150, N) + 1)*N + min(150, N) * (min(150, N) + 9) / 2 + 1)];

		for (int i = 0; i < N; i++)
		{
			X[i] = (double)(i + 1);
			B[i] = (double)(i + 1);
		}

		dfgmres_init(&N, X, B, &RCI_request, ipar, dpar, tmp);
		cout << "Init: RCI_request = " << RCI_request << endl;

		dfgmres_check(&N, X, B, &RCI_request, ipar, dpar, tmp);
		cout << "Init: RCI_request = " << RCI_request << endl;

		dfgmres(&N, X, B, &RCI_request, ipar, dpar, tmp);
		cout << "Init: RCI_request = " << RCI_request << endl;

		cout << "tmp output = [";
		for (int i = 0; i < N; i++)
			cout << tmp[ipar[21] - 1 + i] << " ";
		cout << endl;

		cl.ApplyA(&tmp[ipar[21] - 1], &tmp[ipar[22] - 1], MPI_COMM_WORLD);



		dfgmres(&N, X, B, &RCI_request, ipar, dpar, tmp);
		cout << "Init: RCI_request = " << RCI_request << endl;

		cout << "tmp output = [";
		for (int i = 0; i < N; i++)
			cout << tmp[ipar[21] - 1 + i] << " ";
		cout << endl;

		dfgmres(&N, X, B, &RCI_request, ipar, dpar, tmp);
		cout << "Init: RCI_request = " << RCI_request << endl;

		cout << "Itercount = " << itercount << endl;
		dfgmres_get(&N, X, B, &RCI_request, ipar, dpar, tmp, &itercount);
		cout << "Init: RCI_request = " << RCI_request << endl;
		cout << "Itercount = " << itercount << endl;

		cout << "X = [";
		for (int i = 0; i < N; i++)
			cout << X[i] << " ";
		cout << endl;
	}
	// ==============================

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	if (rank == 0) cout << "Time Elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
	return 0;
}