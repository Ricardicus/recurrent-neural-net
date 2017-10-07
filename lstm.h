#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "utilities.h"
#include "set.h"
#include "layers.h"
#include "assert.h"


#define NEURONS												512

#define STD_LEARNING_RATE										0.01
#define STD_MOMENTUM											0.0
#define STD_LAMBDA											0.05
#define SOFTMAX_TEMP											1.0
#define GRADIENT_CLIP_LIMIT										5.0
#define MINI_BATCH_SIZE											10
#define LOSS_MOVING_AVG											0.01

#define TWO_LAYERS
// #define ONE_LAYER

#define GRADIENTS_CLIP
// #define GRADIENTS_FIT

#define DECREASE_LR 

// #define MODEL_REGULARIZE

// #define DEBUG_PRINT

#define STD_LEARNING_RATE_DECREASE								500000
#define STD_LEARNING_RATE_THRESHOLD								10000

// #define STORE_DURING_TRANING
#define PRINT_EVERY_X_ITERATIONS								200
#define STORE_EVERY_X_ITERATIONS								8000
#define STORE_PROGRESS_EVERY_X_ITERATIONS						1000

#define NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING				100

#define YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE				1
#define YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE		0

#define STD_LOADABLE_NET_NAME									"lstm_net.net"
#define PROGRESS_FILE_NAME										"progress.csv"

typedef struct lstm_model_parameters_t {
	// For progress monitoring
	double loss_moving_avg;
	// For gradient descent
	double learning_rate;
	double momentum;
	double lambda;
	double softmax_temp;

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

typedef struct lstm_values_next_cache_t {
	double* dldh_next;
	double* dldc_next;
	double* dldY_pass;
} lstm_values_next_cache_t;

//					 F,   N,  &lstm model,    zeros, parameters
int lstm_init_model(int, int, lstm_model_t**, int, lstm_model_parameters_t *);
void lstm_zero_the_model(lstm_model_t*);
void lstm_zero_d_next(lstm_values_next_cache_t *, int);
void lstm_cache_container_set_start(lstm_values_cache_t *);

//					 lstm model to be freed
void lstm_free_model(lstm_model_t*);
//							model, input,  state and cache values, &state and cache values
void lstm_forward_propagate(lstm_model_t*, double*, lstm_values_cache_t*, lstm_values_cache_t*, int);
//							model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
void lstm_backward_propagate(lstm_model_t*, double*, int, lstm_values_next_cache_t*, lstm_values_cache_t*, lstm_model_t*, lstm_values_next_cache_t*);

lstm_values_cache_t*  lstm_cache_container_init(int N, int F);
void lstm_cache_container_free(lstm_values_cache_t*);
void lstm_values_next_cache_init(lstm_values_next_cache_t**, int, int);
void lstm_values_next_cache_free(lstm_values_next_cache_t*);
void sum_gradients(lstm_model_t*, lstm_model_t*);

// For storing and loading of net data
//					model (already init), name
void lstm_store_net(lstm_model_t*, const char *);
void lstm_read_net(lstm_model_t*, const char *);
void lstm_store_net_two_layers(lstm_model_t*,lstm_model_t*, const char *);
void lstm_read_net_two_layers(lstm_model_t*,lstm_model_t*, const char *);
void lstm_store_progress(unsigned int, double);

// The main entry point
//						model, number of training points, X_train, Y_train, number of iterations
void lstm_train_the_net(lstm_model_t*, set_T*, unsigned int, int*, int*, unsigned long);
void lstm_train_the_net_two_layers(lstm_model_t*, lstm_model_t*, lstm_model_t*, set_T*, unsigned int, int*, int*, unsigned long);
// Used to output a given number of characters from the net based on an input char
void lstm_output_string(lstm_model_t *, set_T*, char, int);
void lstm_output_string_from_string(lstm_model_t *, set_T*, char *, int); 
void lstm_output_string_two_layers(lstm_model_t *,lstm_model_t *, set_T*, char, int);
void lstm_output_string_from_string_two_layers(lstm_model_t *, lstm_model_t *, set_T*, char *, int);

#endif
