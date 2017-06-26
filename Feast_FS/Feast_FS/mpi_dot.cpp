#include <mpi.h>
#include <iostream>
using namespace std;

double* local_v1;
double* local_v2;
double* local_output;
double* sums;
double local_sum;

void mpi_dot(double* v1, double* v2, double* final_sum, int N, int rank, int size, MPI_Comm comm)
{
	// Initialize local data
	local_v1 = new double[N / size];
	local_v2 = new double[N / size];
	if (rank == 0)
	{
		sums = new double[size];
	}
	local_sum = 0.0;

	// Scatter v1 and v2 across processors
	MPI_Scatter(v1, N / size, MPI_DOUBLE, local_v1, N / size, MPI_DOUBLE, 0, comm);
	MPI_Scatter(v2, N / size, MPI_DOUBLE, local_v2, N / size, MPI_DOUBLE, 0, comm);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
		local_sum += local_v1[i] * local_v2[i];

	// Gather local sums
	MPI_Gather(&local_sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, 0, comm);

	// Compute final sum
	if (rank == 0)
		for (int i = 0; i < size; i++)
			*final_sum += sums[i];

	// Clean up
	delete[] local_v1;
	delete[] local_v2;
	delete[] sums;
}
void mpi_plus(double* v1, double* v2, double* output, int N, int rank, int size, MPI_Comm comm, bool plusorminus)
{
	// Initialize local data
	local_v1 = new double[N / size];
	local_v2 = new double[N / size];
	local_output = new double[N / size];

	// Scatter v1 and v2 across processors
	MPI_Scatter(v1, N / size, MPI_DOUBLE, local_v1, N / size, MPI_DOUBLE, 0, comm);
	MPI_Scatter(v2, N / size, MPI_DOUBLE, local_v2, N / size, MPI_DOUBLE, 0, comm);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
	{
		if (plusorminus)
			local_output[i] = local_v1[i] + local_v2[i];
		else
			local_output[i] = local_v1[i] - local_v2[i];
	}

	// Gather local sums
	MPI_Gather(local_output, N / size, MPI_DOUBLE, output, N / size, MPI_DOUBLE, 0, comm);

	// Clean up
	delete[] local_v1;
	delete[] local_v2;
	delete[] local_output;
}
void mpi_times(double* v, double* output, double scalar, int N, int rank, int size, MPI_Comm comm)
{
	// Initialize local data
	local_v1 = new double[N / size];
	local_output = new double[N / size];

	// Scatter v1 and v2 across processors
	MPI_Scatter(v, N / size, MPI_DOUBLE, local_v1, N / size, MPI_DOUBLE, 0, comm);

	// Compute local sum
	for (int i = 0; i < N / size; i++)
		local_output[i] = local_v1[i] * scalar;

	// Gather local sums
	MPI_Gather(local_output, N / size, MPI_DOUBLE, output, N / size, MPI_DOUBLE, 0, comm);

	// Clean up
	delete[] local_v1;
	delete[] local_output;
}