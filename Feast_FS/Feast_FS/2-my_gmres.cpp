#include <iostream>
#include <time.h>
#include <mpi.h>
#include <cmath>
#include <complex>
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


	// ============================
	// == Main Execution Section ==
	// ============================

	// Seed RNG
	if (rank == 0) srand(time(NULL));
	
	// Declare Variables
	const int d = 2;
	const int N = d*d*d;
	const int m = 6;
	const int L = 1;
	complex<double>** V;
	complex<double>* H;
	complex<double>* r0;
	complex<double>* x0;
	complex<double>* w;
	complex<double>* b;
	complex<double>* temp1;
	complex<double>* e1;
	complex<double>* Hy;
	complex<double>* y;
	complex<double>* Hcopy;
	complex<double> alpha = 0.0f;
	complex<double> beta = 0.0f;
	complex<double> dot_sum = 0.0f;
	complex<double> temp_val;

	// Allocate Memory
	V = new complex<double>*[m + 1];
	for (int i = 0; i < m + 1; i++)
	{
		V[i] = new complex<double>[N];
	}
	H = new complex<double>[(m + 1)*m]();
	Hcopy = new complex<double>[(m + 1)*m]();
	r0 = new complex<double>[N];
	w = new complex<double>[N];
	b = new complex<double>[N];
	x0 = new complex<double>[N];
	e1 = new complex<double>[m + 1]();
	y = new complex<double>[m]();
	temp1 = new complex<double>[N];

	// Initialize
	CubeFD cl(d, d, d, rank, size, MPI_COMM_WORLD);
	for (int i = 0; i < N; i++)
	{
		b[i] = (double)(i + 1);
		x0[i] = (double)rand() / (double)RAND_MAX;
	}

	// for l = 1:L
	for (int l = 0; l < L; l++)
	{
		// r0 = b - A*x0;
		cl.ApplyB(x0, r0, 0.1f, MPI_COMM_WORLD);
		mpi_plus(b, r0, r0, N, rank, size, MPI_COMM_WORLD, false);

		// beta = norm(r0,2);
		dot_sum = 0.0f;
		mpi_dot(r0, r0, &dot_sum, N, rank, size, MPI_COMM_WORLD);
		beta = sqrt(dot_sum);

		// V(1,:) = 1.0/beta * r0;
		mpi_times(r0, V[0], 1.0f / beta, N, rank, size, MPI_COMM_WORLD);

		// 1) Compute Upper Hessenberg Matrix H_k using Arnoldi process

		// for j = 1:m
		for (int j = 0; j < m; j++)
		{
			// w = A * V(j,:)';
			cl.ApplyB(V[j], w, 0.1, MPI_COMM_WORLD);

			// for i = 1:j
			for (int i = 0; i <= j; i++)
			{
				// H(i, j) = sqrt(w' * w);
				dot_sum = 0.0f;
				mpi_dot(w, V[i], &dot_sum, N, rank, size, MPI_COMM_WORLD);
				H[i*m + j] = sqrt(dot_sum);
				Hcopy[i*m + j] = H[i*m + j];

				// w = w - H(i,j)*V(i,:)';
				mpi_times(V[i], temp1, H[i*m + j], N, rank, size, MPI_COMM_WORLD);
				mpi_plus(w, temp1, w, N, rank, size, MPI_COMM_WORLD, false);
			}

			// H(j+1, j) = norm(w,2);
			dot_sum = 0.0f;
			mpi_dot(w, w, &dot_sum, N, rank, size, MPI_COMM_WORLD);
			H[(j + 1)*m + j] = sqrt(dot_sum);
			Hcopy[(j + 1)*m + j] = H[(j + 1)*m + j];

			// V(j+1, :) = (1.0 / H(j+1, j) * w)';
			mpi_times(w, V[j + 1], 1.0f / H[(j + 1)*m + j], N, rank, size, MPI_COMM_WORLD);

		}

		// 2) Solve Linear Least Squares Problem ||beta * e1 - H_l * y_hat|| using LAPACKE_sgels
		//		lapack_int LAPACKE_sgels (int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, float* a, lapack_int lda, float* b, lapack_int ldb);
		e1[0] = beta;
		y[0] = beta;

		if (l == 0)
		{
			cout << "[";
			for (int i = 0; i < m + 1; i++)
			{
				for (int j = 0; j < m; j++)
				{
					cout << "\t" << H[i*m + j];
				}
				cout << endl;
			}
			cout << "]\n";
		}

		lapack_int info = LAPACKE_zgels(LAPACK_ROW_MAJOR, 'N', m + 1, m, 1, H, m, y, 1);

		if (info != 0) cout << "Something went wrong with the least squares computation!\n";

		// 3) Form Approximate Solution
		for (int i = 0; i < m; i++)
		{
			mpi_times(V[i], temp1, y[i], N, rank, size, MPI_COMM_WORLD);
			mpi_plus(x0, temp1, x0, N, rank, size, MPI_COMM_WORLD, true);
		}

		// 4) Check Convergence
		//		||beta * e1 - H_l * y_hat|| < epsilon
		// ============================

		cblas_zgemv(CblasRowMajor, CblasNoTrans, m + 1, m, 1.0f, Hcopy, m, y, 1, 0.0f, temp1, 1);

		mpi_plus(e1, temp1, temp1, m + 1, rank, size, MPI_COMM_WORLD, false);

		dot_sum = 0.0f;
		mpi_dot(temp1, temp1, &dot_sum, m + 1, rank, size, MPI_COMM_WORLD);

		cout << "(l = " << l << ") ||beta * e1 - H_l * y_hat|| = " << sqrt(dot_sum) << endl;

		if (l == L - 1)
		{
			if (N < 20)
			{
				cout << "x0 = [";
				for (int i = 0; i < N; i++)
					cout << x0[i] << " ";
				cout << "]\n";
			}
			else
			{
				cout << "x0 = [";
				for (int i = 0; i < 10; i++)
					cout << x0[i] << " ";
				cout << "\n...\n";
				for (int i = 0; i < 10; i++)
					cout << x0[N - 10 + i] << " ";
				cout << "]\n";
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
	if (rank == 0) cout << "Time Elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
	return 0;
}