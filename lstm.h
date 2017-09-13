#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include "utilities.h"
#include "set.h"
#include "layers.h"

#define STD_LEARNING_RATE										0.0005

#define NEURONS													128

#define GRADIENT_CLIP_LIMIT										5

#define MINI_BATCH_SIZE											12

#define PRINT_EVERY_X_ITERATIONS								200
#define STORE_EVERY_X_ITERATIONS								200
#define STORE_PROGRESS_EVERY_X_ITERATIONS						5000

#define NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING				100

#define YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE				1
#define YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBER_PLEASE		0

#define LOSS_MOVING_AVG											0.001

#define STD_LOADABLE_NET_NAME									"lstm_net.net"
#define PROGRESS_FILE_NAME										"progress.csv"

typedef struct lstm_model_t
{
		int F; // Number of features
		int N; // Number of neurons
		int S; // The sum of the above..

		double learning_rate; // for gradient decend..

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

} lstm_model_t;

typedef struct lstm_values_cache_t {
	double* probs;
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

typedef struct lstm_values_next_cache_t {
	double* dldh_next;
	double* dldc_next;
} lstm_values_next_cache_t;

//					 F,   N,  &lstm model,    zeros
int lstm_init_model(int, int, lstm_model_t**, int);
void lstm_zero_the_model(lstm_model_t*);
void lstm_zero_d_next(lstm_values_next_cache_t *);
void lstm_cache_container_set_start(lstm_values_cache_t *);

//					 lstm model to be freed
void lstm_free_model(lstm_model_t*);
//							model, input,  state and cache values, &state and cache values
void lstm_forward_propagate(lstm_model_t*, int, lstm_values_cache_t*, lstm_values_cache_t*);
//							model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
void lstm_backward_propagate(lstm_model_t*, double*, int, lstm_values_next_cache_t*, lstm_values_cache_t*, lstm_model_t*, lstm_values_next_cache_t*);

lstm_values_cache_t*  lstm_cache_container_init(int N, int F);
void lstm_cache_container_free(lstm_values_cache_t*);
void lstm_values_next_cache_init(lstm_values_next_cache_t**, int);
void lstm_values_next_cache_free(lstm_values_next_cache_t*);
void sum_gradients(lstm_model_t*, lstm_model_t*);

// For storing and loading of net data
//					model (already init), name
void lstm_store_net(lstm_model_t*, const char *);
void lstm_read_net(lstm_model_t*, const char *);
void lstm_store_progress(unsigned int, double);

// The main entry point
//						model, number of training points, X_train, Y_train, number of iterations
void lstm_train_the_next(lstm_model_t*, set_T*, unsigned int, int*, int*, unsigned long);
// Used to output a given number of characters from the net based on an input char
void lstm_output_string(lstm_model_t *, set_T*, char, int);

#endif