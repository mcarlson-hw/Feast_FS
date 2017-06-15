#pragma once
#include <mpi.h>
#include <complex>
class CubeFD
{
private:
	// Constants
	const complex<double> PI = 3.1415927;
	const complex<double> C = -0.008443431966;
	int* periods;

	// Data
	complex<double>* local_in;
	complex<double>* local_out;
	complex<double>* top_neighbor;
	complex<double>* bottom_neighbor;
	complex<double>* left_neighbor;
	complex<double>* right_neighbor;
	complex<double>* front_neighbor;
	complex<double>* back_neighbor;
	complex<double>* top_data;
	complex<double>* bottom_data;
	complex<double>* left_data;
	complex<double>* right_data;
	complex<double>* front_data;
	complex<double>* back_data;

	// Parameters
	int n_rows, n_cols, n_layers, n_elems;

	complex<double> hx, hy, hz, h;
	complex<double> a, ax, ay, az, ah;
	int* divs;

	// MPI Stuff
	int p_id, n_processors, cart_rank;
	MPI_Comm cart_comm;
	int p_up, p_down, p_left, p_right, p_front, p_back;
	int* p_XYZ;
	MPI_Request up_r, down_r, left_r, right_r, front_r, back_r;
	MPI_Request up_s, down_s, left_s, right_s, front_s, back_s;

public:
	int* IJK;


	// Constructors
	CubeFD(int, int, int, int, int, MPI_Comm);

	// Internal Functions
	void ApplyA(complex<double>*, complex<double>*, MPI_Comm comm);
	void ApplyB(complex<double>*, complex<double>*, complex<double>, MPI_Comm comm);
	void ApplyC(complex<double>*, complex<double>*, complex<double>, MPI_Comm comm);
	void ApplyM(complex<double>*, complex<double>*, MPI_Comm comm);
	void PrepareOutgoingBuffers();

	// Static Functions
	void set_divs(int);

	// Coordinate Functions
	int ijk_to_m(int, int, int);
	int jk_to_m(int, int);
	int ij_to_m(int, int);
	int ik_to_m(int, int);
	void m_to_ijk(int);

	// MPI
	void parallel_init(MPI_Comm);
	void communicate();
	void wait_for_sends();
	void wait_for_recvs();

	// Destructor
	//~CubeMesh();  // Free memory upon destruction
};