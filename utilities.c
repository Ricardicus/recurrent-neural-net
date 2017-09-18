#include "utilities.h"
/* 
* Dealing with common vector operations 
*
* ==================== WARNING ====================
* 	The caller should have thought about the memory 
*	   allocation, these functions assumes that 
*	   everything is OK. If not used with care, 
*		prohibted reads/writes might occur.
* =================================================
*
*/

extern int debug_print_on;

// used on contigous vectors
void 	vectors_add(double* A, double* B, int L)
{
	int l = 0;
	while ( l < L ) {
		A[l] += B[l];
		++l;
	}
}

// A = A + (B * s)
void 	vectors_add_scalar_multiply(double* A, double* B, int L, double s)
{
	int l = 0;
	while ( l < L ) {
		A[l] += B[l] * s;
		++l;
	}
}

void 	vectors_substract(double* A, double* B, int L)
{
	int l = 0;
	while ( l < L ) {
		A[l] -= B[l];
		++l;
	}
}
// A = A - (B * s)
void 	vectors_substract_scalar_multiply(double* A, double* B, int L, double s)
{
	int l = 0;
	while ( l < L ) {
		A[l] -= B[l] * s;
		++l;
	}
}


void 	vectors_multiply(double* A, double* B, int L)
{
	int l = 0;
	while ( l < L ) {
		A[l] *= B[l];
		++l;
	}
}
void 	vectors_mutliply_scalar(double* A, double b, int L)
{
	int l = 0;
	while ( l < L ) {
		A[l] *= b;
		++l;
	}
}	

int 	init_random_matrix(double*** A, int R, int C)
{
	int r = 0, c = 0;

	*A = calloc(R, sizeof(double*));

	if ( *A == NULL )
		return -1;

	while ( r < R ) {
		(*A)[r] = calloc(C, sizeof(double));
		if ( (*A)[r] == NULL )
			return -2;
		++r;
	}

	r = 0, c = 0;

	while ( r < R ){
		c = 0;
		while ( c < C ){
			(*A)[r][c] =  sample_normal() / sqrt( R * C / 2.0 ); 
			++c;
		}
		++r;
	}

	return 0;
}

double*		get_random_vector(int L, int R) {
	
	int l = 0;
	double *p;
	p = calloc(L, sizeof(double));
	if ( p == NULL )
		exit(0);

	while ( l < L ){
		p[l] = sample_normal() / sqrt( R / 2.0 );
		++l;
	}

	return p;

}

double** 	get_random_matrix(int R, int C)
{
	int r = 0, c = 0;
	double ** p;
	p = calloc(R, sizeof(double*));

	if ( p == NULL )
		exit(-1);

	while ( r < R ) {
		p[r] = calloc(C, sizeof(double));
		if ( p[r] == NULL )
			exit(-1);
		++r;
	}

	r = 0, c = 0;

	while ( r < R ){
		c = 0;
		while ( c < C ){
			p[r][c] =  ((( (double) rand() ) / RAND_MAX) ) / sqrt( R / 2.0 ); 
			++c;
		}
		++r;
	}

	return p;
}

double** 	get_zero_matrix(int R, int C)
{
	int r = 0, c = 0;
	double ** p;
	p = calloc(R, sizeof(double*));

	if ( p == NULL )
		exit(-1);

	while ( r < R ) {
		p[r] = calloc(C, sizeof(double));
		if ( p[r] == NULL )
			exit(-1);
		++r;
	}

	r = 0, c = 0;

	while ( r < R ){
		c = 0;
		while ( c < C ){
			p[r][c] =  0.0;
			++c;
		}
		++r;
	}

	return p;
}

int 	init_zero_matrix(double*** A, int R, int C)
{
	int r = 0, c = 0;

	*A = calloc(R, sizeof(double*));

	if ( *A == NULL )
		return -1;

	while ( r < R ) {
		(*A)[r] = calloc(C, sizeof(double));
		if ( (*A)[r] == NULL )
			return -2;
		++r;
	}

	r = 0, c = 0;

	while ( r < R ){
		c = 0;
		while ( c < C ){
			(*A)[r][c] = 0.0;
			++c;
		}
		++r;
	}

	return 0;
}

int 	free_matrix(double** A, int R)
{
	int r = 0;
	while ( r < R ) {
		free(A[r]);
		++r;	
	}
	free(A);
	return 0;
}

int 	init_zero_vector(double** V, int L) 
{
	int l = 0;
	*V = calloc(L, sizeof(double));
	if ( *V == NULL )
		return -1;

	while ( l < L ){
		(*V)[l] = 0.0;
		++l;
	}
 
	return 0;
}

double* 	get_zero_vector(int L) 
{
	int l = 0;
	double *p;
	p = calloc(L, sizeof(double));
	if ( p == NULL )
		exit(0);

	while ( l < L ){
		p[l] = 0.0;
		++l;
	}

	return p;
}

int 	free_vector(double** V)
{
	free(*V);
	*V = NULL;
	return 0;
}

void 	copy_vector(double* A, double* B, int L)
{
	int l = 0;

	while ( l < L ){
		A[l] = B[l];
		++l;
	}
}

void 	matrix_add(double** A, double** B, int R, int C)
{
	int r = 0, c = 0;

	while ( r < R ) {
		c = 0;
		while ( c < C ) {
			A[r][c] += B[r][c];
			++c;
		}
		++r;
	}
}

void 	vector_set_to_zero(double* V, int L )
{
	int l = 0;
	while ( l < L )
		V[l++] = 0.0;
}


void 	matrix_set_to_zero(double** A, int R, int C)
{
	int r = 0, c = 0;

	while ( r < R ) {
		c = 0;
		while ( c < C ) {
			A[r][c] = 0.0;
			++c;
		}
		++r;
	}
}

void 	matrix_substract(double** A, double** B, int R, int C)
{
	int r = 0, c = 0;

	while ( r < R ) {
		c = 0;
		while ( c < C ) {
			A[r][c] -= B[r][c];
			++c;
		}
		++r;
	}
}

void 	matrix_scalar_multiply(double** A, double b, int R, int C)
{
	int r = 0, c = 0;

	while ( r < R ) {
		c = 0;
		while ( c < C ) {
			A[r][c] *= b;
			++c;
		}
		++r;
	}
}
void 	matrix_clip(double** A, double limit, int R, int C)
{
	int r = 0, c = 0;

	while ( r < R ) {
		c = 0;
		while ( c < C ) {
			if ( A[r][c] > limit )
				A[r][c] = limit;
			else if ( A[r][c] < -limit )
				A[r][C] = -limit;
			++c;
		}
		++r;
	}
}

double one_norm(double* V, int L)
{
	int l = 0;
	double norm = 0.0;
	while ( l < L ) {
		norm += fabs(V[l]);
		++l;
	}

	return norm;
}

int 	vectors_fit(double* V, double limit, int L)
{
	int l = 0;
	int msg = 0;
	double norm;
	while ( l < L ){
		if ( V[l] > limit || V[l] < -limit ){
			msg = 1;
			norm = one_norm(V, L);
			break;
		}
		++l;
	}

	if ( msg )
		vectors_mutliply_scalar(V, limit / norm, L);

	return msg;
}

int 	vectors_clip(double* V, double limit, int L)
{
	int l = 0;
	int msg = 0;
	while ( l < L ){
		if ( V[l] > limit ){
			msg = 1;
			V[l] = limit;
		} else if(  V[l] < -limit ){
			msg = 1;
			V[l] = -limit;
			l = 0;
		}
		++l;
	}

	return msg;
}

void 	matrix_store(double ** A, int R, int C, FILE * fp) 
{
	int r = 0, c = 0;
	size_t i = 0;
	void * p;

	while ( r < R ) {
		c = 0;
		while ( c < C ) {
			i = 0; p = &A[r][c];
			while ( i < sizeof(double) ) {
				fputc(*((char*)p), fp);
				++i; ++p;
			}
			++c;
		}
		++r;
	}

}

void 	matrix_read(double ** A, int R, int C, FILE * fp) 
{
	int r = 0, c = 0;
	size_t i = 0;
	void * p;
	double value;

	while ( r < R ) {
		c = 0;
		while ( c < C ) {
			i = 0; p = &value;
			while ( i < sizeof(double) ) {
				*((char *)p) = fgetc(fp);
				++i; ++p;
			}
			A[r][c] = value;
			++c;
		}
		++r;
	}

}

void 	vector_store(double* V, int L, FILE * fp)
{
	int l = 0;
	size_t i = 0;
	void * p;

	while ( l < L ){
		i = 0; p = &V[l];
		while ( i < sizeof(double)) {
			fputc(*((char*)p), fp);
			++i; ++p;
		}
		++l;
	}
}

void 	vector_read(double * V, int L, FILE * fp) 
{
	int l = 0;
	size_t i = 0;
	void * p;
	double value;

	while ( l < L ) {
		i = 0; p = &value;
		while ( i < sizeof(double) ) {
			*((char *)p) = fgetc(fp);
			++i; ++p;
		}
		V[l] = value;
		++l;
	}

}

double sample_normal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sample_normal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}