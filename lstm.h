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
#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "utilities.h"
#include "set.h"
#include "layers.h"
#include "assert.h"

#define	OPTIMIZE_ADAM											0
#define OPTIMIZE_GRADIENT_DESCENT								1

typedef struct lstm_model_parameters_t {
	// For progress monitoring
	double loss_moving_avg;
	// For gradient descent
	double learning_rate;
	double momentum;
	double lambda;
	double softmax_temp;
	double beta1;
	double beta2;
	int gradient_clip;
	int gradient_fit;
	int optimizer;
	int model_regularize;
	int stateful;
	int decrease_lr;

	// How many layers
	int layers;

	// Output configuration for interactivity
	long print_progress_iterations;
	int  print_progress_sample_output;
	int  print_progress;
	int  print_progress_to_file;
	int  print_progress_number_of_chars;
	char *print_sample_output_to_file_name;
	char *print_sample_output_to_file_arg;
	int  store_progress_evert_x_iterations;
	char *store_progress_file_name;

	int learning_rate_decrease_threshold;
	double learning_rate_decrease;
	// General parameters
	int mini_batch_size;
	double gradient_clip_limit;
} lstm_model_parameters_t;

typedef struct lstm_model_t
{
		int F; // Number of features
		int N; // Number of neurons
		int S; // The sum of the above..

		// Parameters
		lstm_model_parameters_t * params;

		// The model
		double* Wf;
		double* Wi;
		double* Wc;
		double* Wo;
		double* Wy;
		double* bf;
		double* bi;
		double* bc;
		double* bo;
		double* by;

		// cache
		double* dldh;
		double* dldho;
		double* dldhf;
		double* dldhi;
		double* dldhc;
		double* dldc;

		double* dldXi;
		double* dldXo;
		double* dldXf;
		double* dldXc;

		// Gradient descent momentum
		double* Wfm;
		double* Wim;
		double* Wcm;
		double* Wom;
		double* Wym;
		double* bfm;
		double* bim;
		double* bcm;
		double* bom;
		double* bym;


} lstm_model_t;

typedef struct lstm_values_cache_t {
	double* probs;
	double* probs_before_sigma;
	double* c;
	double* h;
	double* c_old;
	double* h_old;
	double* X;
	double* hf;
	double* hi;
	double* ho;
	double* hc;
	double* tanh_c_cache;
} lstm_values_cache_t;

typedef struct lstm_values_state_t {
	double* c;
	double* h;
} lstm_values_state_t;

typedef struct lstm_values_next_cache_t {
	double* dldh_next;
	double* dldc_next;
	double* dldY_pass;
} lstm_values_next_cache_t;

//					 F,   N,  &lstm model,    zeros, parameters
int lstm_init_model(int, int, lstm_model_t**, int, lstm_model_parameters_t *);
void lstm_zero_the_model(lstm_model_t*);
void lstm_zero_d_next(lstm_values_next_cache_t *, int, int);
void lstm_cache_container_set_start(lstm_values_cache_t *, int);

//					 lstm model to be freed
void lstm_free_model(lstm_model_t*);
//							model, input,  state and cache values, &state and cache values
void lstm_forward_propagate(lstm_model_t*, double*, lstm_values_cache_t*, lstm_values_cache_t*, int);
//							model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
void lstm_backward_propagate(lstm_model_t*, double*, int, lstm_values_next_cache_t*, lstm_values_cache_t*, lstm_model_t*, lstm_values_next_cache_t*);

void lstm_values_state_init(lstm_values_state_t** d_next_to_set, int N);
void lstm_values_next_state_free(lstm_values_state_t* d_next);

lstm_values_cache_t*  lstm_cache_container_init(int N, int F);
void lstm_cache_container_free(lstm_values_cache_t*);
void lstm_values_next_cache_init(lstm_values_next_cache_t**, int, int);
void lstm_values_next_cache_free(lstm_values_next_cache_t*);
void sum_gradients(lstm_model_t*, lstm_model_t*);

// For storing and loading of net data
//					model (already init), name
void lstm_read_net_layers(lstm_model_t**, const char*, int);
void lstm_store_net_layers(lstm_model_t**, const char *, int);
void lstm_store_net_layers_as_json(lstm_model_t**, const char *, const char *, set_T *, int);
void lstm_store_progress(const char*, unsigned int, double);

// The main entry point
//						model, number of training points, X_train, Y_train, number of iterations
void lstm_train(lstm_model_t*, lstm_model_t**, set_T*, unsigned int, int*, int*, unsigned long, int);
// Used to output a given number of characters from the net based on an input char
void lstm_output_string_layers(lstm_model_t **, set_T*, int, int, int);
void lstm_output_string_from_string_layers(lstm_model_t **, set_T*, char *, int, int);

#endif
