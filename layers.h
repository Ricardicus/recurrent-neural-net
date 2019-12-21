/*
* This file is part of the LSTM Network implementation In C made by Rickard Hallerbäck
* 
*                 Copyright (c) 2018 Rickard Hallerbäck
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), 
* to deal in the Software without restriction, including without limitation the rights 
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
* Software, and to permit persons to whom the Software is furnished to do so, subject to 
* the following conditions:
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
*
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
* OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <stdlib.h>
#include <math.h>

// Dealing with FC layers, forward and backward
//		Y = AX + b  			&Y,      A,   		X,		B,     Rows (A), Columns (A)
void fully_connected_forward(double*, double*, double*, double*, int, int);
//		Y = AX + b  			dldY,       A,     X,        &dldA,    &dldX,    &dldb   Rows (A), Columns (A)
void fully_connected_backward(double*, double*, double* ,double*, double*, double*, int,    int);

// Dealing with softmax layer, forward and backward
//								&P,		Y,  	features
void softmax_layers_forward(double*, double*, int, double);
//									  P,	  c,  &dldh, rows
void softmax_loss_layer_backward(double*, int, double*, int);

// Other layers used: sigmoid and tanh
// 	
// 		Y = sigmoid(X), &Y, X, length
void sigmoid_forward(double *, double*, int);
// 		Y = sigmoid(X), dldY, Y, &dldX, length
void sigmoid_backward(double *, double*, double*, int);
// 		Y = tanh(X), &Y, X, length
void tanh_forward(double *, double*, int);
// 		Y = tanh(X), dldY, Y, &dldX, length
void tanh_backward(double *, double*, double*, int);

// The loss function used
double cross_entropy(double*, int);