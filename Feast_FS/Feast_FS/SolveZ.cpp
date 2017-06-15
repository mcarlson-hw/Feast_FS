#include <iostream>
#include <complex>
#include "mpi.h"
#include "CubeFD.h"
#include "mpi_dot.h"
#include "mkl_types.h"
#define MKL_Complex16 std::complex<double>
#include "mkl.h"
using namespace std;
void SolveZ(complex<double> Ze, int m, complex<double> beta, complex<double>** H_ptrs,
			complex<double>** V, complex<double>** Z, complex<double>* H, 
			complex<double>* w, complex<double>* b, complex<double>* y, complex<double>* y_ptrs,
			complex<double>* e1, complex<double>* x0, complex<double>* r0, complex<double>* temp1,
			int L, int N, int rank, int size, MPI_Comm comm, CubeFD cl)
{
	complex<double> temp_val;
	complex<double>* v_ptr;
	complex<double>* z_ptr;
	complex<double> one = (complex<double>) 1.0;
	complex<double> zero = (complex<double>) 0.0;

	// for l = 1:L
	for (int l = 0; l < L; l++)
	{
		// r0 = b - A*x0;
		cl.ApplyZ(x0, r0, Ze, MPI_COMM_WORLD);
		mpi_plus(b, r0, r0, N, rank, size, MPI_COMM_WORLD, false);

		// beta = norm(r0,2);
		mpi_dot(r0, r0, &beta, N, rank, size, MPI_COMM_WORLD);
		if (rank == 0) beta = sqrt(beta);

		// V(:,1) = 1.0/beta * r0;
		if (rank == 0) temp_val = (complex<double>)1.0 / beta;
		if (rank == 0) v_ptr = V[0];
		mpi_times(r0, v_ptr, temp_val, N, rank, size, MPI_COMM_WORLD);

		// for j = 1:m
		for (int j = 0; j < m; j++)
		{
			if (rank == 0) v_ptr = V[j];
			if (rank == 0) z_ptr = Z[j];

			// w = A * V(j,:)';
			cl.ApplyZc(v_ptr, z_ptr, Ze, MPI_COMM_WORLD);
			cl.ApplyZ(z_ptr, w, Ze, MPI_COMM_WORLD);

			//		for i = 1:j
			for (int i = 0; i <= j; i++)
			{
				if (rank == 0) v_ptr = V[i];
				//	H(i, j) = sqrt(w' * w);
				mpi_dot(w, v_ptr, &H_ptrs[i][j], N, rank, size, MPI_COMM_WORLD);
				if (rank == 0) H_ptrs[i][j] = sqrt(H_ptrs[i][j]);

				//	w = w - H(i,j)*V(:,i);
				if (rank == 0) temp_val = H_ptrs[i][j];
				mpi_times(v_ptr, temp1, temp_val, N, rank, size, MPI_COMM_WORLD);
				mpi_plus(w, temp1, w, N, rank, size, MPI_COMM_WORLD, false);
			}
			//		H(j+1, j) = norm(w,2);
			mpi_dot(w, w, &H_ptrs[j + 1][j], N, rank, size, MPI_COMM_WORLD);
			if (rank == 0) H_ptrs[j + 1][j] = sqrt(H_ptrs[j + 1][j]);

			//		V(:, j+1) = (1.0 / H(j+1, j) * w);
			if (rank == 0) temp_val = (complex<double>) 1.0 / H_ptrs[j + 1][j];
			if (rank == 0) v_ptr = V[j + 1];
			mpi_times(w, v_ptr, temp_val, N, rank, size, MPI_COMM_WORLD);
		}

		// Solve Least Squares Problem
		if (rank == 0)
		{
			for (int k = 0; k < m + 1; k++)
			{
				e1[k] = (complex<double>) 0.0;
				y_ptrs[k] = (complex<double>) 0.0;
			}
			e1[0] = beta;
			y_ptrs[0] = beta;

			lapack_int info = LAPACKE_zgels(LAPACK_ROW_MAJOR, 'N', m + 1, m, 1, H, m, y, 1);

			//cout << "y = [";
			//for (int k = 0; k < m; k++)
			//	cout << y[k][0] << " ";
			//cout << "]\n";

			if (info != 0) cout << "Something went wrong with the least squares computation!\n";
		}

		// Form Approximate Solution
		for (int i = 0; i < m; i++)
		{
			if (rank == 0) z_ptr = Z[i];
			if (rank == 0) temp_val = y[i];
			mpi_times(z_ptr, temp1, temp_val, N, rank, size, MPI_COMM_WORLD);
			mpi_plus(x0, temp1, x0, N, rank, size, MPI_COMM_WORLD, true);
		}

		// Check Residual
		if (rank == 0) cblas_zgemv(CblasRowMajor, CblasNoTrans, m + 1, m, &one, H, m, y, 1, &zero, temp1, 1);
		mpi_plus(e1, temp1, temp1, m + 1, rank, size, MPI_COMM_WORLD, false);
		mpi_dot(temp1, temp1, &beta, m + 1, rank, size, MPI_COMM_WORLD);
		if (rank == 0) cout << "(l = " << l << ") ||beta * e1 - H_l * y_hat|| = " << sqrt(beta) << endl;

		// Output final approximate solution
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
}