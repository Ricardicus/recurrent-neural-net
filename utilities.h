#ifndef LSTM_UTILITIES_H
#define LSTM_UTILITIES_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// used on contigous vectors
//		A = A + B		A,		B,    l
void 	vectors_add(double*, double*, int);
void 	vectors_substract(double*, double*, int);
void 	vectors_add_scalar_multiply(double*, double*, int, double);
void 	vectors_substract_scalar_multiply(double*, double*, int, double);
//		A = A + B		A,		B,    R, C
void 	matrix_add(double**, double**, int, int);
void 	matrix_substract(double**, double**, int, int);
//		A = A*b		A,		b,    R, C
void 	matrix_scalar_multiply(double**, double, int, int);

//		A = A * B		A,		B,    l
void 	vectors_multiply(double*, double*, int);
//		A = A * b		A,		b,    l
void 	vectors_mutliply_scalar(double*, double, int);
//		A = random( (R, C) ) / sqrt(R / 2), &A, R, C
int 	init_random_matrix(double***, int, int);
//		A = 0.0s, &A, R, C
int 	init_zero_matrix(double***, int, int);
int 	free_matrix(double**, int);
//						 V to be set, Length
int 	init_zero_vector(double**, int);
int 	free_vector(double**);
//		A = B       A,		B,		length
void 	copy_vector(double*, double*, int);
double* 	get_zero_vector(int); 
double** 	get_zero_matrix(int, int);
double** 	get_random_matrix(int, int);
double* 	get_random_vector(int,int);

void 	matrix_set_to_zero(double**, int, int);
void 	vector_set_to_zero(double*, int);

void matrix_clip(double**, double, int, int);
void vectors_fit(double*, double, int);

// I/O
void 	vector_read(double *, int, FILE *);
void 	vector_store(double *, int, FILE *);
void 	matrix_store(double **, int, int, FILE *);  
void 	matrix_read(double **, int, int, FILE *);   
#endif