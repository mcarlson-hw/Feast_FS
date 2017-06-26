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

	// Seed RNG
	if (rank == 0) srand(time(NULL));

	// Declare Variables
	const int d = 20;
	const int N = d*d*d;
	const int m = 6;
	const int L = 1000;
	complex<double>** V;
	complex<double>** Z;
	complex<double> H[m + 1][m];
	complex<double> Hcopy[m + 1][m];
	complex<double>* r0;
	complex<double>* x0;
	complex<double>* w;
	complex<double>* b;
	complex<double>* temp1;
	complex<double>* e1;
	complex<double>* Hy;
	complex<double> y[m + 1][1];
	complex<double> alpha = 0.0f;
	complex<double> beta = 0.0f;
	complex<double> dot_sum = 0.0f;
	complex<double> temp_val;
	complex<double> one = (complex<double>) 1.0;
	complex<double> zero = (complex<double>) 0.0;
	complex<double>* v_ptr;
	complex<double>* z_ptr;
	complex<double> Ze;

	Ze.real(6.49242);
	Ze.imag(-4.698);
	// Allocate Memory
	if (rank == 0)
	{
		V = new complex<double>*[m + 1];
		Z = new complex<double>*[m + 1];
		for (int i = 0; i < m + 1; i++)
		{
			V[i] = new complex<double>[N];
			Z[i] = new complex<double>[N];
			y[i][0] = (complex<double>) 0.0;
			if (rank == 0)
			{
				for (int j = 0; j < m; j++)
				{
					H[i][j] = (complex<double>) 0.0;
					Hcopy[i][j] = (complex<double>) 0.0;
				}
			}
		}

		r0 = new complex<double>[N];
		w = new complex<double>[N];
		b = new complex<double>[N];
		x0 = new complex<double>[N];
		e1 = new complex<double>[m + 1]();
		temp1 = new complex<double>[N];
	}

	// Initialize
	CubeFD cl(d, d, d, rank, size, MPI_COMM_WORLD);

	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			b[i] = (complex<double>)(i + 1);
			x0[i] = b[i];
		}
	}

	// for l = 1:L
	for (int l = 0; l < L; l++)
	{
		// r0 = b - A*x0;
		cl.ApplyZ(x0, r0, Ze, MPI_COMM_WORLD);
		mpi_plus(b, r0, r0, N, rank, size, MPI_COMM_WORLD, false);

		// beta = norm(r0,2);
		dot_sum = (complex<double>) 0.0;
		mpi_dot(r0, r0, &dot_sum, N, rank, size, MPI_COMM_WORLD);
		if (rank == 0) beta = sqrt(dot_sum);

		// V(1,:) = 1.0/beta * r0;
		if (rank == 0) temp_val = (complex<double>)1.0 / beta;
		else temp_val = zero;
		if (rank == 0) v_ptr = V[0];
		else v_ptr = NULL;
		mpi_times(r0, v_ptr, temp_val, N, rank, size, MPI_COMM_WORLD);

		// 1) Compute Upper Hessenberg Matrix H_k using Arnoldi process

		// for j = 1:m
		for (int j = 0; j < m; j++)
		{
			if (rank == 0) v_ptr = V[j];
			else v_ptr = NULL;
			if (rank == 0) z_ptr = Z[j];
			else v_ptr = NULL;

			// w = A * V(j,:)';
			cl.ApplyZc(v_ptr, z_ptr, Ze, MPI_COMM_WORLD);
			cl.ApplyZ(z_ptr, w, Ze, MPI_COMM_WORLD);

			// for i = 1:j
			for (int i = 0; i <= j; i++)
			{
				if (rank == 0) v_ptr = V[i];
				else v_ptr = NULL;

				// H(i, j) = sqrt(w' * w);
				dot_sum = (complex<double>) 0.0;
				mpi_dot(w, v_ptr, &dot_sum, N, rank, size, MPI_COMM_WORLD);
				if (rank == 0) H[i][j] = sqrt(dot_sum);
				if (rank == 0) Hcopy[i][j] = H[i][j];

				// w = w - H(i,j)*V(i,:)';
				if (rank == 0) temp_val = H[i][j];
				else temp_val = zero;

				mpi_times(v_ptr, temp1, temp_val, N, rank, size, MPI_COMM_WORLD);
				mpi_plus(w, temp1, w, N, rank, size, MPI_COMM_WORLD, false);

			}

			// H(j+1, j) = norm(w,2);
			dot_sum = (complex<double>) 0.0;
			mpi_dot(w, w, &dot_sum, N, rank, size, MPI_COMM_WORLD);
			if (rank == 0) H[j + 1][j] = sqrt(dot_sum);
			if (rank == 0) Hcopy[j + 1][j] = H[j + 1][j];

			// V(j+1, :) = (1.0 / H(j+1, j) * w)';
			if (rank == 0) temp_val = (complex<double>) 1.0 / H[j + 1][j];
			else temp_val = zero;
			if (rank == 0) v_ptr = V[j + 1];
			else v_ptr = NULL;
			mpi_times(w, v_ptr, temp_val, N, rank, size, MPI_COMM_WORLD);

		}

		// 2) Solve Linear Least Squares Problem ||beta * e1 - H_l * y_hat|| using LAPACKE_sgels
		//		lapack_int LAPACKE_sgels (int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, float* a, lapack_int lda, float* b, lapack_int ldb);
		if (rank == 0)
		{
			for (int k = 0; k < m + 1; k++)
			{
				e1[k] = (complex<double>) 0.0;
				y[k][0] = (complex<double>) 0.0;
			}
			e1[0] = beta;
			y[0][0] = beta;

			lapack_int info = LAPACKE_zgels(LAPACK_ROW_MAJOR, 'N', m + 1, m, 1, *H, m, *y, 1);

			//cout << "y = [";
			//for (int k = 0; k < m; k++)
			//	cout << y[k][0] << " ";
			//cout << "]\n";

			if (info != 0) cout << "Something went wrong with the least squares computation!\n";

		}

		// 3) Form Approximate Solution
		for (int i = 0; i < m; i++)
		{
			if (rank == 0) z_ptr = Z[i];
			else z_ptr = NULL;
			if (rank == 0) temp_val = y[i][0];
			else temp_val = zero;
			mpi_times(z_ptr, temp1, temp_val, N, rank, size, MPI_COMM_WORLD);
			mpi_plus(x0, temp1, x0, N, rank, size, MPI_COMM_WORLD, true);
		}

		// 4) Check Convergence
		//		||beta * e1 - H_l * y_hat|| < epsilon
		// ============================

		if (rank == 0) cblas_zgemv(CblasRowMajor, CblasNoTrans, m + 1, m, &one, *H, m, *y, 1, &zero, temp1, 1);

		mpi_plus(e1, temp1, temp1, m + 1, rank, size, MPI_COMM_WORLD, false);

		dot_sum = (complex<double>) 0.0;
		mpi_dot(temp1, temp1, &dot_sum, m + 1, rank, size, MPI_COMM_WORLD);

		if (rank == 0) cout << "(l = " << l << ") ||beta * e1 - H_l * y_hat|| = " << sqrt(dot_sum) << endl;

		if (l == L - 1 && rank == 0)
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