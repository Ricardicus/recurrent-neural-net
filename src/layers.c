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

/* 
* Dealing with FC layers, forward and backward
*
* ==================== NOTE: ====================
*   The caller should have thought about the memory 
*    allocation, these functions assumes that 
*           everything is OK. 
* =================================================
*
*/

#include "layers.h"

#ifdef WINDOWS
#include <stdio.h>
#endif

//    Y = AX + b        &Y,      A,       X,    B,     Rows (for A), Columns (for A)
void  fully_connected_forward(double* Y, double* A, double* X, double* b, int R, int C)
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
//    Y = AX + b        dldY,       A,     X,        &dldA,    &dldX,    &dldb   Rows (A), Columns (A)
void  fully_connected_backward(double* dldY, double* A, double* X,double* dldA,
  double* dldX, double* dldb, int R, int C)
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

double cross_entropy(double* probabilities, int correct)
{
  return -log(probabilities[correct]);  
}

// Dealing with softmax layer, forward and backward
//                &P,   Y,    features
void  softmax_layers_forward(double* P, double* Y, int F, double temperature)  
{
  int f = 0;
  double sum = 0;
#ifdef WINDOWS
  // MSVC is not a C99 compiler, and does not support variable length arrays
  // MSVC is documented as conforming to C90
  double *cache = malloc(sizeof(double)*F);

  if ( cache == NULL ) {
    fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n", 
      __FILE__, __func__, __LINE__, sizeof(double)*F);
    exit(1);
  }
#else
  double cache[F];
#endif

  while ( f < F ) {
    cache[f] = exp(Y[f] / temperature);
    sum += cache[f];
    ++f;
  }

  f = 0;
  while ( f < F ) {
    P[f] = cache[f] / sum;
    ++f;
  }

#ifdef WINDOWS
  free(cache);
#endif
}
//                    P,    c,  &dldh, rows
void  softmax_loss_layer_backward(double* P, int c, double* dldh, int R)
{ 
  int r = 0;

  while ( r < R ) {
    dldh[r] = P[r];
    ++r;
  }

  dldh[c] -= 1.0;
}
// Other layers used: sigmoid and tanh
//  
//    Y = sigmoid(X), &Y, X, length
void  sigmoid_forward(double* Y, double* X, int L)
{
  int l = 0;

  while ( l < L ) {
    Y[l] = 1.0 / ( 1.0 + exp(-X[l]));
    ++l;
  }

}
//    Y = sigmoid(X), dldY, Y, &dldX, length
void  sigmoid_backward(double* dldY, double* Y, double* dldX, int L) 
{
  int l = 0;

  while ( l < L ) {
    dldX[l] = ( 1.0 - Y[l] ) * Y[l] * dldY[l];
    ++l;
  }

}
//    Y = tanh(X), &Y, X, length
void  tanh_forward(double* Y, double* X, int L)
{
  int l = 0;
  while ( l < L ) {
    Y[l] = tanh(X[l]);
    ++l;
  }
}
//    Y = tanh(X), dldY, Y, &dldX, length
void  tanh_backward(double* dldY, double* Y, double* dldX, int L)
{
  int l = 0;
  while ( l < L ) {
    dldX[l] = ( 1.0 - Y[l] * Y[l] ) * dldY[l];
    ++l;
  }
}
