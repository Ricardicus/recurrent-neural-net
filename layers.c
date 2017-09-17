#include "layers.h"
/* 
* Dealing with FC layers, forward and backward
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

//		Y = AX + b  			&Y,      A,   		X,		B,     Rows (for A), Columns (for A)
void	fully_connected_forward(double* Y, double* A, double* X, double* b, int R, int C)
{
	int i = 0, n = 0;
	while ( i < R ) {
		Y[i] = b[i];
		n = 0;
		while ( n < C ) {
			Y[i] += A[i * C + n] * X[n];
			++n;
		}
		++i;
	}

}
//		Y = AX + b  			dldY,       A,     X,        &dldA,    &dldX,    &dldb   Rows (A), Columns (A)
void 	fully_connected_backward(double* dldY, double* A, double* X,double* dldA, double* dldX, double* dldb, int R, int C)
{
	int i = 0, n = 0;

	// computing dldA
	while ( i < R ) {
		n = 0;
		while ( n < C ) {
			dldA[i * C + n] = dldY[i] * X[n];
			++n;
		}
		++i;
	}

	// computing dldb (easy peasy)
	i = 0;
	while ( i < R ) {
		dldb[i] = dldY[i];
		++i;
	}

	// computing dldX 
	i = 0, n = 0;
	while ( i < C ) {
		n = 0;

		dldX[i] = 0.0;
		while ( n < R ) {
			dldX[i] += A[n * C + i] * dldY[n];
			++n;
		}

		++i;
	}
}

double cross_entropy(double* probs, int correct)
{
	return -log(probs[correct]);	
}

// Dealing with softmax layer, forward and backward
//								&P,		Y,  	features
void 	softmax_layers_forward(double* P, double* Y, int F)  
{
	int f = 0;
	double sum = 0;
	double cache[F];

	while ( f < F ) {
		cache[f] = exp(Y[f]);
		sum += cache[f];
		++f;
	}

	f = 0;
	while ( f < F ) {
		P[f] = cache[f] / sum;
		++f;
	}
}
//									  P,	  c,  &dldh, rows
void 	softmax_loss_layer_backward(double* P, int c, double* dldh, int R)
{	
	int r = 0;

	while ( r < R ){
		dldh[r] = P[r];
		++r;
	}

	dldh[c] -= 1.0;
}
// Other layers used: sigmoid and tanh
// 	
// 		Y = sigmoid(X), &Y, X, length
void 	sigmoid_forward(double* Y, double* X, int L)
{
	int l = 0;

	while ( l < L ) {
		Y[l] = 1.0 / ( 1.0 + exp(-X[l]));
		++l;
	}

}
// 		Y = sigmoid(X), dldY, Y, &dldX, length
void 	sigmoid_backward(double* dldY, double* Y, double* dldX, int L) 
{
	int l = 0;

	while ( l < L ) {
		dldX[l] = ( 1.0 - Y[l] ) * Y[l] * dldY[l];
		++l;
	}

}
// 		Y = tanh(X), &Y, X, length
void 	tanh_forward(double* Y, double* X, int L)
{
	int l = 0;
	while ( l < L ){
		Y[l] = tanh(X[l]);
		++l;
	}
}
// 		Y = tanh(X), dldY, Y, &dldX, length
void 	tanh_backward(double* dldY, double* Y, double* dldX, int L)
{
	int l = 0;
	while ( l < L ) {
		dldX[l] = ( 1.0 - Y[l] * Y[l] ) * dldY[l];
		++l;
	}
}
