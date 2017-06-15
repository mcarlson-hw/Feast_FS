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

	// Feast Variables
	MKL_INT info = -5;
	MKL_INT ijob = -1;
	MKL_INT M;
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

	// GMRES variables
	const int m = 6;
	const int L = 200;
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
	complex<double> temp_val;

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

		// GMRES stuff
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

		// Call one round of dfeast_srci
		dfeast_srci(&ijob, &N, &ze, work, workc, aq, sq, fpm, &epsout, &loop, &emin, &emax, &m0, lambda, q, &M, res, &info);

		cout << "ijob = " << ijob << endl;
		cout << "ze = " << ze << endl;
		cout << "info = " << info << endl;
		cout << "m = " << m << endl;

		dfeast_srci(&ijob, &N, &ze, work, workc, aq, sq, fpm, &epsout, &loop, &emin, &emax, &m0, lambda, q, &m, res, &info);

		cout << "ijob = " << ijob << endl;
		cout << "ze = " << ze << endl;
		cout << "info = " << info << endl;
		cout << "m = " << m << endl;

		// Solve (ZeB - A)x = workc
		SolveZ(Ze, m, beta, H_ptrs, V, Z, *H, w, b, *y, y_ptrs, e1, x0, r0, temp1, L, N, rank, size, MPI_COMM_WORLD, cl);
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