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
	const int d = 50;
	const MKL_INT N = d*d*d;
	double* X;
	double* B;
	MKL_INT RCI_request;
	MKL_INT* ipar;
	double* dpar;
	double* tmp;
	MKL_INT itercount = 0;
	int itercount2 = 0;
	double* tmp1;
	double* tmp2;
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

		// Initialize GMRES (do once)
		dfgmres_init(&N, X, B, &RCI_request, ipar, dpar, tmp);
		cout << "Init: RCI_request = " << RCI_request << endl;
		ipar[4] = 600;

		// Check parameters for consistency (not required unless parameters manually changed)
		dfgmres_check(&N, X, B, &RCI_request, ipar, dpar, tmp);
		cout << "Check: RCI_request = " << RCI_request << endl;
	}

	while (itercount2 < 2000)
	{
		if (rank == 0) dfgmres(&N, X, B, &RCI_request, ipar, dpar, tmp);
		MPI_Bcast(&RCI_request, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (RCI_request == 1)
		{
			if (rank == 0)
			{
				tmp1 = &tmp[ipar[21] - 1];
				tmp2 = &tmp[ipar[22] - 1];
			}
			cl.ApplyA(tmp1, tmp2, MPI_COMM_WORLD);
		}
		if (RCI_request == 2)
		{
			// Do nothing for now
		}
		if (RCI_request == 3)
		{
			// Do nothing for now
		}
		if (RCI_request == 4)
		{
			if (rank == 0)
			{
				if (dpar[6] < 1e-11)
				{
					cout << "gmres has converged.\n";
					cout << "dpar[6] = " << dpar[6] << endl;
					dfgmres_get(&N, X, B, &RCI_request, ipar, dpar, tmp, &itercount);
					itercount2 = 1000000;
					break;
				}
				else
				{
					cout << "dpar[6] = " << dpar[6] << endl;
				}
			}
		}
		if (RCI_request == 0)
		{
			if (rank == 0)
			{
				cout << "gmres has completed but did not converge to expected tolerance.\n";
				dfgmres_get(&N, X, B, &RCI_request, ipar, dpar, tmp, &itercount);
			}
			itercount2 = 1000000;
			break;
		}
		if (RCI_request < 0)
		{
			if (rank == 0)
			{
				cout << "Something has gone wrong!\n";
				cout << "RCI_request = " << RCI_request << endl;
			}
		}
		itercount2++;
	}
	if (rank == 0)
	{
		if (RCI_request == 0)
		{
			if (N < 100)
			{
				cout << "X = [ ";
				for (int i = 0; i < N; i++)
					cout << X[i] << " ";
				cout << "]\n";
			}
			else
			{
				cout << "x = [ ";
				for (int i = 0; i < 10; i++)
					cout << X[i] << " ";
				cout << "\n...\n";
				for (int i = 0; i < 10; i++)
					cout << X[N - 10 + i] << " ";
				cout << "]\n";
			}
		}
	}
	// ==============================

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	if (rank == 0) cout << "Time Elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
	return 0;
}