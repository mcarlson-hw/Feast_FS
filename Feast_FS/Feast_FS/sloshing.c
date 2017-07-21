#include "mgm_matrix.h"
#include "mgm_cnodes.h"
#include <mangll_cvec.h>
#include <mangll_cnodes.h>
#include <mangll_tensor.h>
#include <sc_dmatrix.h>
#include <sc_flops.h>
#include <p8est.h>
#include <mangll_p8est.h>
#include <mangll_mesh.h>
#include <mkl.h>

#include "ymir_trilinear_elem.h"

#include "colors.h"

void solve_fs_eigs(mangll_domain_t *domain, mangll_locidx_t num_cnodes,
									 	int num_fields);
void my_zmgres(MKL_Complex16 Ze, MKL_Complex16* workc);

int
main(int argc, char **argv)
{
	// Variables: MPI
	int                 rank, num_procs, mpiret;
	MPI_Comm            mpicomm = MPI_COMM_WORLD;

	// for perf.
	sc_flopinfo_t       flopinfo;

	// Variables: mangll
	int                 order_basis_fcts = 2;
	int                 forest_new_level = 4;
	mangll_domain_t    *domain;
	mangll_mesh_t      *mesh;
	mangll_t           *mangll;
	mangll_locidx_t     num_local_elmts;
	mangll_locidx_t     num_nodes_per_el; /* number of nodes per element*/
	mangll_locidx_t     num_cnodes, num_own_cnodes, own_offset;

	int                 num_grids = 1;
	int                 num_jac = 10;
	double              jac_alpha = 6.0 / 7.0;

	// Variables: work
	int                 num_fields = 1;
	int                 i, j, k;

	if (argc < 4) {
		printf(RED "Usage: %s forest_level num_grids jacobi_iter jacobi_rel\n"
			NRM, argv[0]);
		return -1;
	}
	forest_new_level = atoi(argv[1]);
	num_grids = atoi(argv[2]);
	num_jac = atoi(argv[3]);
	// jac_alpha = atof ( argv[4] );

	// initialize MPI
	mpiret = MPI_Init(&argc, &argv);
	SC_CHECK_MPI(mpiret);
	mpiret = MPI_Comm_size(mpicomm, &num_procs);
	SC_CHECK_MPI(mpiret);
	mpiret = MPI_Comm_rank(mpicomm, &rank);
	SC_CHECK_MPI(mpiret);

	// initialize packages
	sc_init(mpicomm, 1, 1, NULL, SC_LP_SILENT);
#ifdef MANGLL_WITH_P4EST
	p4est_init(NULL, SC_LP_PRODUCTION);
#endif
	mangll_init(NULL, SC_LP_DEFAULT);

	// create new mangll domain structure
	domain = mangll_domain_new(mpicomm);
	mangll_domain_set_paramsf(domain, "domain", "forest_unitcube",
																		"forest_new_level", forest_new_level,
																		"refine", "uniform", NULL);
	mesh = mangll_domain_setup_mesh(domain);

	// partition with coarsening correction
	p8est_partition_ext(domain->p8est, 1, NULL);
	p8est_ghost_destroy(domain->ghost);
	domain->ghost = p8est_ghost_new(domain->p8est, P8EST_CONNECT_FULL);
	mangll_mesh_destroy(domain->mesh);
	domain->mesh = mangll_p8est_mesh_new_full(domain->p8est, domain->ghost);
	domain->mesh->X_fn = domain->X_fn;
	domain->mesh->X_data = domain;

	// create new mangll inside domain
	mangll = mangll_domain_setup_mangll(domain, order_basis_fcts - 1);

	// get mangll variables
	num_local_elmts = mesh->K;
	num_nodes_per_el = mangll->Np;
	num_cnodes = domain->cnodes->Ncn;
	num_own_cnodes = domain->cnodes->owncount;
	own_offset = 0;



	// ==========================================================================
	// ==========================================================================
	// == My code ===============================================================
	// ==========================================================================
	// ==========================================================================

	solve_fs_eigs(domain, num_cnodes, num_fields);

	// ==========================================================================
	// ==========================================================================


	// destroy mangll
	mangll_domain_destroy(domain);

	// clean up and exit
	sc_finalize();

	mpiret = MPI_Finalize();
	SC_CHECK_MPI(mpiret);

	return 0;
}

void solve_fs_eigs(mangll_domain_t *domain, mangll_locidx_t num_cnodes,
										int num_fields)
{
	MANGLL_GLOBAL_PRODUCTION(GRN "FS: Solving Eigenvalue Problem\n" NRM);

	int i, count;

	// Feast Declarations
	MKL_INT* fpm;
	MKL_INT ijob = -1;
	MKL_INT N = num_cnodes*num_fields;
	MKL_Complex16 ze;
	double* work;
	MKL_Complex16* workc;
	double* aq;
	double* sq;
	double epsout;
	MKL_INT loop = 0;
	const double emin = 0.1;
	const double emax = 10.0;
	MKL_INT m0 = 10;
	double* lambda;
	double* q;
	MKL_INT m;
	double* res;
	MKL_INT info = -5;

	// Feast Allocations
	fpm = malloc(sizeof(MKL_INT) * 128);
	double _res[m0][1];
	double _lambda[m0][1];
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

	feastinit(fpm);
	dfeast_srci(&ijob, &N, &ze, work, workc, aq, sq, fpm, &epsout, &loop, &emin,
								&emax, &m0, lambda, q, &m, res, &info);
	for (count = 0; count < 10; count++)
	{
		switch (ijob)
		{
		case 10:  printf("Case 10 reached.\n");
			break;
		case 11:  printf("Case 11 reached.\n");
			break;
		case 30:  printf("Case 30 reached.\n");
			break;
		case 40:  printf("Case 40 reached.\n");
			break;
		case 0:   printf("Case 0 reached.\n");
			break;
		default:  printf("Default case reached.\n");
			break;
		}
		dfeast_srci(&ijob, &N, &ze, work, workc, aq, sq, fpm, &epsout, &loop,
									&emin, &emax, &m0, lambda, q, &m, res, &info);
	}

	// create matrices
	sc_dmatrix_t *u_in, *u_out;
	double* ones;
	ones = malloc(sizeof(double)*num_cnodes*num_fields);

	for (i = 0; i < num_cnodes*num_fields; i++)
		ones[i] = 1.0;

	u_in = sc_dmatrix_new_data(num_cnodes, num_fields, ones);
	u_out = sc_dmatrix_new(num_cnodes, num_fields);

	apply_mass_matrix(domain, u_in, u_out);

	printf("The number of entries is %d\n", num_cnodes*num_fields);

	double* test_data = *(u_out->e);
	printf("[");
	for (i = 0; i < num_cnodes*num_fields; i++)
		printf("%f ", test_data[i]);
	printf("]\n");

	sc_dmatrix_destroy(u_in);
	sc_dmatrix_destroy(u_out);
}

// Solve ZeB - A = workc
void my_zmgres(MKL_Complex16 Ze, MKL_Complex16* workc, int N, int M, int L)
{
	int i, j, l;
	MKL_Complex16 beta, betainv;
	// z_j	1..m      sc_dmatrix_t(N, 1) (complex) times (m+1)
	// v_j  1..m+1		sc_dmatrix_t(N, 1) (complex) times (m+1)
	sc_dmatrix_t *V_real = malloc(sizeof(sc_dmatrix_t*)*(m+1));
	sc_dmatrix_t *V_imag = malloc(sizeof(sc_dmatrix_t*)*(m+1));
	for (i = 0; i < m+1; i++)
	{
		V_real[i] = sc_matrix_new(N, 1);
		V_imag[i] = sc_matrix_new(N, 1);
	}
	// w							sc_dmatrix_t(N, 1) (complex)
	sc_dmatrix_t w_real = sc_dmatrix_new(N, 1);
	sc_dmatrix_t w_imag = sc_dmatrix_new(N, 1);
	// h_i_j					double _Complex
	MKL_Complex16 H[M+1][M];
	// r0							sc_dmatrix_t(N, 1) (complex)
	sc_dmatrix_t r0_real = sc_dmatrix_new(N, 1);
	sc_dmatrix_t r0_imag = sc_dmatrix_new(N, 1);

	// 	turn workc into a sc_dmatrix_t
	sc_dmatrix_t b_real = sc_dmatrix_new(N, 1);
	sc_dmatrix_t b_imag = sc_dmatrix_new(N, 1);
	sc_dmatrix_t x0_real = sc_dmatrix_new(N, 1);
	sc_dmatrix_t x0_imag = sc_dmatrix_new(N, 1);
	for (i = 0; i < N; i++)
	{
		b_real->e[i][0] = workc[i].re;
		b_imag->e[i][0] = workc[i].im;
		x0_real->e[i][0] = workc[i].re;
		x0_imag->e[i][0] = workc[i].im;
	}

	//Y := alpha X + Y
	//sc_dmatrix_add (double alpha, const sc_dmatrix_t * X, sc_dmatrix_t * Y);
  // FOR l = 1..L
	for (l = 0; l < L; l++)
	{
		//	compute r0 = b - Ax0		first iteration is r0 = workc - A*work
		apply_stiffness_matrix (x0_real, r0_real);
		apply_stiffness_matrix (x0_imag, r0_imag);
		sc_dmatrix_scale (-1.0, r0_real);
		sc_dmatrix_add (1.0, b_real, r0_real);
		sc_dmatrix_scale (-1.0, r0_imag);
		sc_dmatrix_add (1.0, b_imag, r0_imag);
		//  beta = sqrt( (r0, r0) )
		beta = complex_inner_product(r0_real, r0_imag, r0_real, r0_imag);
		betainv.re = +1.0 * beta.re / (beta.re*beta.re + beta.im*beta.im);
		betainv.im = -1.0 * beta.im / (beta.re*beta.re + beta.im*beta.im);
		// 	v1 = r0 / beta					complex scalar product
		complex_scalar_product(N, betainv, r0_real[0], r0_imag[0],
																			 V_real[0],  V_imag[0]);
		//	FOR j = 1..M
		for (j = 0; j < M; j++)
		{
			//		z_j = preconditioned(v_j)
			//		w = K*z_j							if no preconditioner, w = K*v_j
			apply_stiffness_matrix
			//		FOR i = 1..j
			//			H_i_j = ( w, v_i )
			//			w = w - h_i_j * v_i
			//		h_(j+1)_j = sqrt( (w, w) )
			//		v_(j+1) = w / h_(j+1)_j
		}

		// 	with H and beta computed, solve least squares problem
		// 	form approximate solution
		// 	compute reduced residual and check for stopping criteria
		//	otherwise x0 = x and repeat
	}

}

MKL_Complex16 complex_inner_product(sc_dmatrix_t a, sc_dmatrix_t b,
																		sc_dmatrix_t c, sc_dmatrix_t d)
{
	// x = a + I*b
	// y = c + I*d
	// (x, y) = x' * y
	//      z = (a, c) + (b, d) + I*((a, d) - (b, c))
	//			z_real = a'*c + b'*d
	//			z_imag = a'*d - b'*c
	sc_dmatrix_t z_real = sc_dmatrix_new(1,1);
	sc_dmatrix_multiply(SC_TRANS, SC_NO_TRANS, 1.0, a, c, 0.0, z_real);
	sc_dmatrix_multiply(SC_TRANS, SC_NO_TRANS, 1.0, b, d, 1.0, z_real);
	sc_dmatrix_t z_imag = sc_dmatrix_new(1,1);
	sc_dmatrix_multiply(SC_TRANS, SC_NO_TRANS, 1.0, a, d, 0.0, z_imag);
	sc_dmatrix_multiply(SC_TRANS, SC_NO_TRANS, 1.0, b, c, -1.0, z_imag);
	MKL_Complex16 retval;
	retval.re = z_real->e[0][0];
	retval.im = z_imag->e[0][0];
	sc_matrix_destroy(z_real);
	sc_matrix_destroy(z_imag);
	return retval;
}

double double_inner_product(sc_dmatrix_t x, sc_dmatrix_t y)
{
	// z = (x, y)
	// z = x'*y
	// C := alpha * A * B + beta * C
	// sc_dmatrix_multiply (sc_trans_t transa, sc_trans_t transb, double alpha,
	//                      const sc_dmatrix_t * A, const sc_dmatrix_t * B,
	//	  									double beta, sc_dmatrix_t * C);
	sc_dmatrix_t z = sc_dmatrix_new(1, 1);
	sc_dmatrix_multiply (SC_TRANS, SC_NO_TRANS, 1.0, x, y, 0.0, z);
	double retval = z->e[0][0];
	sc_matrix_destroy(z);
	return retval;
}

void complex_scalar_product(MKL_INT N, MKL_Complex16 alpha, sc_dmatrix_t x,
														sc_dmatrix_t y, sc_dmatrix_t x_out,
														sc_dmatrix_t y_out)
{
	// alpha = a + I*b
	// z = x + I*y
	// alpha*z = a*x - b*y + I*(a*y + b*x)
	// real(alpha*z) = a*x - b*y
	// imag(alpha*z) = a*y + b*x
	//Y := alpha X + Y
	//sc_dmatrix_add (double alpha, const sc_dmatrix_t * X, sc_dmatrix_t * Y);
	//sc_dmatrix_copy (const sc_dmatrix_t * X, sc_dmatrix_t * Y);
	sc_dmatrix_add (+1.0 * alpha.re, x, x_out);
	sc_dmatrix_add (-1.0 * alpha.im, y, x_out);
	sc_dmatrix_add (alpha.re, y, y_out);
	sc_dmatrix_add (alpha.im, x, y_out);
}
