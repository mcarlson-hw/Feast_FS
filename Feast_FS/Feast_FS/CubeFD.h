#pragma once
#include <mpi.h>
class CubeFD
{
private:
	// Constants
	const double PI = 3.1415927;
	const double C = -0.008443431966;
	int* periods;

	// Data
	double* local_in;
	double* local_out;
	double* top_neighbor;
	double* bottom_neighbor;
	double* left_neighbor;
	double* right_neighbor;
	double* front_neighbor;
	double* back_neighbor;
	double* top_data;
	double* bottom_data;
	double* left_data;
	double* right_data;
	double* front_data;
	double* back_data;

	// Parameters
	int n_rows, n_cols, n_layers, n_elems;

	double hx, hy, hz, h;
	double a, ax, ay, az, ah;
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
	void ApplyA(double*, double*, MPI_Comm comm);
	void ApplyB(double*, double*, double, MPI_Comm comm);
	void ApplyC(double*, double*, double, MPI_Comm comm);
	void ApplyM(double*, double*, MPI_Comm comm);
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