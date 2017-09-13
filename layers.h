#include <stdlib.h>
#include <math.h>

// Dealing with FC layers, forward and backward
//		Y = AX + b  			&Y,      A,   		X,		B,     Rows (A), Columns (A)
void	fully_connected_forward(double*, double*, double*, double*, int, int);
//		Y = AX + b  			dldY,       A,     X,        &dldA,    &dldX,    &dldb   Rows (A), Columns (A)
void 	fully_connected_backward(double*, double*, double* ,double*, double*, double*, int,    int);

// Dealing with softmax layer, forward and backward
//								&P,		Y,  	features
void 	softmax_layers_forward(double*, double*, int);
//									  P,	  c,  &dldh, rows
void 	softmax_loss_layer_backward(double*, int, double*, int);

// Other layers used: sigmoid and tanh
// 	
// 		Y = sigmoid(X), &Y, X, length
void 	sigmoid_forward(double *, double*, int);
// 		Y = sigmoid(X), dldY, Y, &dldX, length
void 	sigmoid_backward(double *, double*, double*, int);
// 		Y = tanh(X), &Y, X, length
void 	tanh_forward(double *, double*, int);
// 		Y = tanh(X), dldY, Y, &dldX, length
void 	tanh_backward(double *, double*, double*, int);

// The loss function used
double	cross_entropy(double*, int);