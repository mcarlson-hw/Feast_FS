#include <cmath>
#include <iostream>
#include <complex>
#include <time.h>
#include "CubeFD.h"
#include "mpi_dot.h"
using namespace std;
CubeFD::CubeFD(int total_rows, int total_columns, int total_layers, int rank, int P, MPI_Comm comm)
{
	set_divs(P);
	p_id = rank;
	n_processors = P;

	periods = new int[3]();

	n_rows = total_rows / divs[0];
	n_cols = total_columns / divs[1];
	n_layers = total_layers / divs[2];
	n_elems = n_rows*n_cols*n_layers;

	hx = (complex<double>) 1.0 / (complex<double>)(total_rows - 1);
	hy = (complex<double>) 1.0 / (complex<double>)(total_columns - 1);
	hz = (complex<double>) 1.0 / (complex<double>)(total_layers - 1);

	ax = (complex<double>) 1.0 / (hx*hx);
	ay = (complex<double>) 1.0 / (hy*hy);
	az = (complex<double>) 1.0 / (hz*hz);
	a = (complex<double>) 2.0 * (ax + ay + az);

	local_in = new complex<double>[n_elems];
	local_out = new complex<double>[n_elems];

	top_neighbor = new complex<double>[n_rows*n_cols]();
	bottom_neighbor = new complex<double>[n_rows*n_cols]();
	left_neighbor = new complex<double>[n_cols*n_layers]();
	right_neighbor = new complex<double>[n_cols*n_layers]();
	front_neighbor = new complex<double>[n_rows*n_layers]();
	back_neighbor = new complex<double>[n_rows*n_layers]();

	top_data = new complex<double>[n_rows*n_cols]();
	bottom_data = new complex<double>[n_rows*n_cols]();
	left_data = new complex<double>[n_cols*n_layers]();
	right_data = new complex<double>[n_cols*n_layers]();
	front_data = new complex<double>[n_rows*n_layers]();
	back_data = new complex<double>[n_rows*n_layers]();

	IJK = new int[3];
	IJK[0] = -1;
	IJK[1] = -1;
	IJK[2] = -1;

	parallel_init(comm);
}

void CubeFD::ApplyA(complex<double>* in, complex<double>* out, MPI_Comm comm)
{
	MPI_Scatter(in, n_elems, MPI_DOUBLE_COMPLEX, local_in, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
	PrepareOutgoingBuffers();
	communicate();

	int m;
	complex<double> sum;

	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		if (IJK[0] == 0 || IJK[0] == n_rows - 1 || IJK[1] == 0 || IJK[0] == n_cols - 1 || IJK[2] == 0 || IJK[2] == n_layers - 1)
			continue;
		sum = (complex<double>) 0.0;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2]);
		local_out[M] = a*local_in[m] - sum;
	}

	wait_for_recvs();

	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		if (IJK[0] != 0 && IJK[0] != n_rows - 1 && IJK[1] != 0 && IJK[0] != n_cols - 1 && IJK[2] != 0 && IJK[2] != n_layers - 1)
			continue;
		sum = (complex<double>) 0.0;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2]);
		local_out[M] = a*local_in[m] - sum;
	}

	wait_for_sends();

	MPI_Gather(local_out, n_elems, MPI_DOUBLE_COMPLEX, out, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
}
void CubeFD::ApplyB(complex<double>* in, complex<double>* out, complex<double> sigma, MPI_Comm comm)
{
	MPI_Scatter(in, n_elems, MPI_DOUBLE_COMPLEX, local_in, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
	PrepareOutgoingBuffers();
	communicate();

	int m;
	complex<double> sum;

	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		if (IJK[0] == 0 || IJK[0] == n_rows - 1 || IJK[1] == 0 || IJK[0] == n_cols - 1 || IJK[2] == 0 || IJK[2] == n_layers - 1)
			continue;
		sum = (complex<double>) 0.0;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2]);
		local_out[M] = (a - sigma)*local_in[m] - sum;
	}

	wait_for_recvs();

	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		if (IJK[0] != 0 && IJK[0] != n_rows - 1 && IJK[1] != 0 && IJK[0] != n_cols - 1 && IJK[2] != 0 && IJK[2] != n_layers - 1)
			continue;
		sum = (complex<double>) 0.0;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2]);
		local_out[M] = (a - sigma)*local_in[m] - sum;
	}

	wait_for_sends();

	MPI_Gather(local_out, n_elems, MPI_DOUBLE_COMPLEX, out, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
}
void CubeFD::ApplyC(complex<double>* in, complex<double>* out, complex<double> sigma, MPI_Comm comm)
{
	MPI_Scatter(in, n_elems, MPI_DOUBLE_COMPLEX, local_in, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
	for (int i = 0; i < n_elems; i++)
		local_out[i] = ((complex<double>) 1.0 / (a - sigma)) * local_in[i];
	MPI_Gather(local_out, n_elems, MPI_DOUBLE_COMPLEX, out, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
}
void CubeFD::ApplyM(complex<double>* in, complex<double>* out, MPI_Comm comm)
{
	for (int i = 0; i < n_rows*divs[0] * n_cols*divs[1] * n_layers*divs[2]; i++)
	{
		out[i] = in[i];
	}
}
void CubeFD::ApplyZc(complex<double>* in, complex<double>* out, complex<double> Ze, MPI_Comm comm)
{
	MPI_Scatter(in, n_elems, MPI_DOUBLE_COMPLEX, local_in, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
	for (int i = 0; i < n_elems; i++)
		local_out[i] = ((complex<double>) 1.0 / (Ze - a)) * local_in[i];
	MPI_Gather(local_out, n_elems, MPI_DOUBLE_COMPLEX, out, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
}
void CubeFD::ApplyZ(complex<double>* in, complex<double>* out, complex<double> Ze, MPI_Comm comm)
{
	// Z = Ze * B - A
	// N = n_elems * divs[0] * divs[1] * divs[2]
	MPI_Scatter(in, n_elems, MPI_DOUBLE_COMPLEX, local_in, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
	PrepareOutgoingBuffers();
	communicate();

	int m;
	complex<double> sum;

	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		if (IJK[0] == 0 || IJK[0] == n_rows - 1 || IJK[1] == 0 || IJK[0] == n_cols - 1 || IJK[2] == 0 || IJK[2] == n_layers - 1)
			continue;
		sum = (complex<double>) 0.0;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2]);
		local_out[M] = (Ze - a)*local_in[m] + sum;
	}

	wait_for_recvs();

	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		if (IJK[0] != 0 && IJK[0] != n_rows - 1 && IJK[1] != 0 && IJK[0] != n_cols - 1 && IJK[2] != 0 && IJK[2] != n_layers - 1)
			continue;
		sum = (complex<double>) 0.0;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*local_in[m]; }
		else { sum += ax*left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*local_in[m]; }
		else { sum += ay*back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*local_in[m]; }
		else { sum += az*bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2]);
		local_out[M] = (Ze - a)*local_in[m] + sum;
	}

	wait_for_sends();

	MPI_Gather(local_out, n_elems, MPI_DOUBLE_COMPLEX, out, n_elems, MPI_DOUBLE_COMPLEX, 0, comm);
}
void CubeFD::PrepareOutgoingBuffers()
{

	for (int j = 0; j < n_cols; j++)
		for (int i = 0; i < n_rows; i++)
		{
			top_data[ij_to_m(i, j)] = local_in[ijk_to_m(i, j, 0)];
			bottom_data[ij_to_m(i, j)] = local_in[ijk_to_m(i, j, n_layers - 1)];
		}
	for (int k = 0; k < n_layers; k++)
		for (int i = 0; i < n_rows; i++)
		{
			front_data[ik_to_m(i, k)] = local_in[ijk_to_m(i, n_cols - 1, k)];
			back_data[ik_to_m(i, k)] = local_in[ijk_to_m(i, 0, k)];
		}
	for (int k = 0; k < n_layers; k++)
		for (int j = 0; j < n_cols; j++)
		{
			left_data[jk_to_m(j, k)] = local_in[ijk_to_m(0, j, k)];
			right_data[jk_to_m(j, k)] = local_in[ijk_to_m(n_rows - 1, j, k)];
		}
}

void CubeFD::set_divs(int p)
{
	int p_divs[25][3] = { { 1, 1, 1 },
	{ 2, 1, 1 },
	{ 3, 1, 1 },
	{ 2, 2, 1 },
	{ 5, 1, 1 },
	{ 3, 2, 1 },
	{ 7, 1, 1 },
	{ 2, 2, 2 },
	{ 3, 3, 1 },
	{ 5, 2, 1 },
	{ 11, 1, 1 },
	{ 3, 2, 2 },
	{ 13, 1, 1 },
	{ 7, 2, 1 },
	{ 5, 3, 1 },
	{ 4, 2, 2 },
	{ 17, 1, 1 },
	{ 3, 3, 2 },
	{ 5, 2, 2 },
	{ 7, 3, 1 },
	{ 11, 2, 1 },
	{ 23, 1, 1 },
	{ 4, 3, 2 },
	{ 5, 5, 1 },
	{ 13, 2, 1 } };

	this->divs = new int[3];
	this->divs[0] = p_divs[p - 1][0];
	this->divs[1] = p_divs[p - 1][1];
	this->divs[2] = p_divs[p - 1][2];
}

int CubeFD::ijk_to_m(int i, int j, int k)
{
	if (i < 0 || i > n_rows - 1 || j < 0 || j > n_cols - 1 || k < 0 || k > n_layers - 1)
		return -1;
	return (i + j * n_rows + k * n_rows * n_cols);
}
int CubeFD::jk_to_m(int j, int k)
{
	// n_cols by n_layers
	return j + k*n_cols;
}
int CubeFD::ij_to_m(int i, int j)
{
	// n_rows by n_cols
	return i + j*n_rows;
}
int CubeFD::ik_to_m(int i, int k)
{
	// n_rows by n_layers
	return i + k*n_rows;
}
void CubeFD::m_to_ijk(int m)
{
	IJK[0] = m % n_rows;
	IJK[1] = (m / n_rows) % n_cols;
	IJK[2] = m / (n_rows * n_cols);
}

void CubeFD::parallel_init(MPI_Comm comm)
{
	MPI_Cart_create(comm, 3, divs, periods, 0, &cart_comm);
	int local_coords[3] = { 0, 0, 0 };
	MPI_Cart_coords(cart_comm, p_id, 3, local_coords);
	p_XYZ = new int[3];
	p_XYZ[0] = local_coords[0];
	p_XYZ[1] = local_coords[1];
	p_XYZ[2] = local_coords[2];

	p_up = -1;
	p_down = -1;
	p_left = -1;
	p_right = -1;
	p_front = -1;
	p_back = -1;

	int right[3] = { local_coords[0] + 1, local_coords[1], local_coords[2] };
	int left[3] = { local_coords[0] - 1, local_coords[1], local_coords[2] };
	int front[3] = { local_coords[0], local_coords[1] + 1, local_coords[2] };
	int back[3] = { local_coords[0], local_coords[1] - 1, local_coords[2] };
	int up[3] = { local_coords[0], local_coords[1], local_coords[2] + 1 };
	int down[3] = { local_coords[0], local_coords[1], local_coords[2] - 1 };

	if (local_coords[0] + 1 < divs[0]) MPI_Cart_rank(cart_comm, right, &p_right);
	if (local_coords[0] - 1 >= 0) MPI_Cart_rank(cart_comm, left, &p_left);
	if (local_coords[1] + 1 < divs[1]) MPI_Cart_rank(cart_comm, front, &p_front);
	if (local_coords[1] - 1 >= 0) MPI_Cart_rank(cart_comm, back, &p_back);
	if (local_coords[2] + 1 < divs[2]) MPI_Cart_rank(cart_comm, up, &p_up);
	if (local_coords[2] - 1 >= 0) MPI_Cart_rank(cart_comm, down, &p_down);
}
void CubeFD::communicate()
{
	if (p_up != -1)
	{
		MPI_Isend(top_data, n_rows*n_cols, MPI_DOUBLE_COMPLEX, p_up, 0, cart_comm, &up_s);
		MPI_Irecv(top_neighbor, n_rows*n_cols, MPI_DOUBLE_COMPLEX, p_up, 0, cart_comm, &up_r);
	}
	if (p_down != -1)
	{
		MPI_Isend(bottom_data, n_rows*n_cols, MPI_DOUBLE_COMPLEX, p_down, 0, cart_comm, &down_s);
		MPI_Irecv(bottom_neighbor, n_rows*n_cols, MPI_DOUBLE_COMPLEX, p_down, 0, cart_comm, &down_r);
	}
	if (p_left != -1)
	{
		MPI_Isend(left_data, n_cols*n_layers, MPI_DOUBLE_COMPLEX, p_left, 0, cart_comm, &left_s);
		MPI_Irecv(left_neighbor, n_cols*n_layers, MPI_DOUBLE_COMPLEX, p_left, 0, cart_comm, &left_r);
	}
	if (p_right != -1)
	{
		MPI_Isend(right_data, n_cols*n_layers, MPI_DOUBLE_COMPLEX, p_right, 0, cart_comm, &right_s);
		MPI_Irecv(right_neighbor, n_cols*n_layers, MPI_DOUBLE_COMPLEX, p_right, 0, cart_comm, &right_r);
	}
	if (p_front != -1)
	{
		MPI_Isend(front_data, n_rows*n_layers, MPI_DOUBLE_COMPLEX, p_front, 0, cart_comm, &front_s);
		MPI_Irecv(front_neighbor, n_rows*n_layers, MPI_DOUBLE_COMPLEX, p_front, 0, cart_comm, &front_r);
	}
	if (p_back != -1)
	{
		MPI_Isend(back_data, n_rows*n_layers, MPI_DOUBLE_COMPLEX, p_back, 0, cart_comm, &back_s);
		MPI_Irecv(back_neighbor, n_rows*n_layers, MPI_DOUBLE_COMPLEX, p_back, 0, cart_comm, &back_r);
	}
}

void CubeFD::wait_for_sends()
{
	if (p_up != -1) MPI_Wait(&up_s, MPI_STATUS_IGNORE);
	if (p_down != -1) MPI_Wait(&down_s, MPI_STATUS_IGNORE);
	if (p_left != -1) MPI_Wait(&left_s, MPI_STATUS_IGNORE);
	if (p_right != -1) MPI_Wait(&right_s, MPI_STATUS_IGNORE);
	if (p_front != -1) MPI_Wait(&front_s, MPI_STATUS_IGNORE);
	if (p_back != -1) MPI_Wait(&back_s, MPI_STATUS_IGNORE);
}
void CubeFD::wait_for_recvs()
{
	if (p_up != -1) MPI_Wait(&up_r, MPI_STATUS_IGNORE);
	if (p_down != -1) MPI_Wait(&down_r, MPI_STATUS_IGNORE);
	if (p_left != -1) MPI_Wait(&left_r, MPI_STATUS_IGNORE);
	if (p_right != -1) MPI_Wait(&right_r, MPI_STATUS_IGNORE);
	if (p_front != -1) MPI_Wait(&front_r, MPI_STATUS_IGNORE);
	if (p_back != -1) MPI_Wait(&back_r, MPI_STATUS_IGNORE);
}