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


	// ============================
	// == Main Execution Section ==
	// ============================

	MKL_INT info = -5;
	MKL_INT ijob = -1;
	MKL_INT m;
	MKL_Complex16 ze;
	double epsout;

	const MKL_INT d = 3;
	const MKL_INT N = d*d*d;
	MKL_INT loop = 0;
	MKL_INT m0 = 10;
	double emin = 0.2;
	double emax = 10.0;

	// 1d array pointers
	MKL_INT* fpm;
	double* res;
	double* lambda;

	// 2d array pointers
	double* work;
	double* aq;
	double* sq;
	double* q;
	MKL_Complex16* workc;

	if (rank == 0)
	{
		// Initialize Feast Parameters
		fpm = new MKL_INT[128];
		feastinit(fpm);

		// Initialize Feast Workspace
		// 1d arrays
		double _res[m0][1];
		double _lambda[m0][1];
		// 2d arrays
		double _work[N][m0];
		double _aq[m0][m0];
		double _sq[m0][m0];
		double _q[N][m];
		MKL_Complex16 _workc[N][m0];

		// Set Pointers
		res = *_res;
		lambda = *_lambda;
		work = *_work;
		aq = *_aq;
		sq = *_sq;
		q = *_q;
		workc = *_workc;

		// Call one round of dfeast_srci
		dfeast_srci(&ijob, &N, &ze, work, workc, aq, sq, fpm, &epsout, &loop, &emin, &emax, &m0, lambda, q, &m, res, &info);

		cout << "ijob = " << ijob << endl;
		cout << "ze = " << ze << endl;
		cout << "info = " << info << endl;
		cout << "m = " << m << endl;

		dfeast_srci(&ijob, &N, &ze, work, workc, aq, sq, fpm, &epsout, &loop, &emin, &emax, &m0, lambda, q, &m, res, &info);

		cout << "ijob = " << ijob << endl;
		cout << "ze = " << ze << endl;
		cout << "info = " << info << endl;
		cout << "m = " << m << endl;
	}

	// void dfeast_srci (	MKL_INT* ijob, const MKL_INT* N, 
	//						MKL_Complex16* ze, double* work, MKL_Complex16* workc, 
	//						double* aq, double* sq, MKL_INT* fpm, double* epsout, 
	//						MKL_INT* loop, const double* emin, const double* emax, 
	//						MKL_INT* m0, double* lambda, double* q, MKL_INT* m, double* res, 
	//						MKL_INT* info);

	// ============================

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	if (rank == 0) cout << "Time Elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
	return 0;
}