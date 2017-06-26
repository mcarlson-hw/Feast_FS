#include <mpi.h>
#include <complex>
#include <iostream>
using namespace std;

complex<double>* local_v1;
complex<double>* local_v2;
complex<double>* local_output;
complex<double>* sums;
complex<double> local_sum;
complex<double> local_scalar;

void mpi_dot(complex<double>* v1, complex<double>* v2, complex<double>* final_sum, int N, int rank, int size, MPI_Comm comm)
{
	// Initialize local data
	local_v1 = new complex<double>[N / size];
	local_v2 = new complex<double>[N / size];
	if (rank == 0)
	{
		sums = new complex<double>[size];
	}
	local_sum = (complex<double>) 0.0;

	// Scatter v1 and v2 across processors
	MPI_Scatter(v1, N / size, MPI_DOUBLE_COMPLEX, local_v1, N / size, MPI_DOUBLE_COMPLEX, 0, comm);
	MPI_Scatter(v2, N / size, MPI_DOUBLE_COMPLEX, local_v2, N / size, MPI_DOUBLE_COMPLEX, 0, comm);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
		local_sum += conj(local_v1[i]) * local_v2[i];

	// Gather local sums
	MPI_Gather(&local_sum, 1, MPI_DOUBLE_COMPLEX, sums, 1, MPI_DOUBLE_COMPLEX, 0, comm);

	// Compute final sum
	if (rank == 0) *final_sum = (complex<double>) 0.0;
	if (rank == 0)
		for (int i = 0; i < size; i++)
			*final_sum += sums[i];

	// Clean up
	delete[] local_v1;
	delete[] local_v2;
	delete[] sums;
}
void mpi_plus(complex<double>* v1, complex<double>* v2, complex<double>* output, int N, int rank, int size, MPI_Comm comm, bool plusorminus)
{
	// Initialize local data
	local_v1 = new complex<double>[N / size];
	local_v2 = new complex<double>[N / size];
	local_output = new complex<double>[N / size];

	// Scatter v1 and v2 across processors
	MPI_Scatter(v1, N / size, MPI_DOUBLE_COMPLEX, local_v1, N / size, MPI_DOUBLE_COMPLEX, 0, comm);
	MPI_Scatter(v2, N / size, MPI_DOUBLE_COMPLEX, local_v2, N / size, MPI_DOUBLE_COMPLEX, 0, comm);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
	{
		if (plusorminus)
			local_output[i] = local_v1[i] + local_v2[i];
		else
			local_output[i] = local_v1[i] - local_v2[i];
	}

	// Gather local sums
	MPI_Gather(local_output, N / size, MPI_DOUBLE_COMPLEX, output, N / size, MPI_DOUBLE_COMPLEX, 0, comm);

	// Clean up
	delete[] local_v1;
	delete[] local_v2;
	delete[] local_output;
}
void mpi_times(complex<double>* v, complex<double>* output, complex<double> scalar, int N, int rank, int size, MPI_Comm comm)
{
	// Initialize local data
	local_v1 = new complex<double>[N / size];
	local_output = new complex<double>[N / size];
	if (rank == 0) local_scalar = scalar;

	// Scatter v1 and v2 across processors
	MPI_Scatter(v, N / size, MPI_DOUBLE_COMPLEX, local_v1, N / size, MPI_DOUBLE_COMPLEX, 0, comm);
	MPI_Bcast(&local_scalar, 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
		local_output[i] = local_v1[i] * local_scalar;

	// Gather local sums
	MPI_Gather(local_output, N / size, MPI_DOUBLE_COMPLEX, output, N / size, MPI_DOUBLE_COMPLEX, 0, comm);

	// Clean up
	delete[] local_v1;
	delete[] local_output;
}