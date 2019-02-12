#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

#include "lstm.h"
#include "set.h"
#include "layers.h"
#include "utilities.h"

#define ITERATIONS 	100000000

lstm_model_t *model = NULL, *layer1 = NULL, *layer2 = NULL;
lstm_model_t **model_layers;
set_T set;

void store_the_net_layers(int signo)
{
	if ( model_layers != NULL ){
		lstm_store_net_layers(model_layers, STD_LOADABLE_NET_NAME);
		lstm_store_net_layers_as_json(model_layers, STD_JSON_NET_NAME, &set);
		printf("\nStored the net as: '%s'\nYou can use that file in the .html interface.\n", 
			STD_JSON_NET_NAME);
		printf("The net in its raw format is stored as: '%s'.\nYou can use that with the -r flag \
to continue refining the weights.\n", STD_LOADABLE_NET_NAME); 
	} else {
		printf("\nFailed to store the net!\n");
		exit(-1);
	}

	exit(0);
	return;
}

int main(int argc, char *argv[])
{
	int i = 0, c, p = 0;
	size_t file_size = 0, sz = 0;
	int *X_train, *Y_train;
	char * clean;
	FILE * fp;

	lstm_model_parameters_t params;
	memset(&params, 0, sizeof(params));

	params.loss_moving_avg = LOSS_MOVING_AVG;
	params.learning_rate = STD_LEARNING_RATE;
	params.momentum = STD_MOMENTUM;
	params.lambda = STD_LAMBDA;
	params.softmax_temp = SOFTMAX_TEMP;
	params.mini_batch_size = MINI_BATCH_SIZE;
	params.gradient_clip_limit = GRADIENT_CLIP_LIMIT;
	params.learning_rate_decrease = STD_LEARNING_RATE_DECREASE;
	params.learning_rate_decrease_threshold = STD_LEARNING_RATE_THRESHOLD;

	params.stateful = STATEFUL;

	params.beta1 = 0.9;
	params.beta2 = 0.999;

	params.gradient_fit = GRADIENTS_FIT;
	params.gradient_clip = GRADIENTS_CLIP;
	params.decrease_lr = DECREASE_LR;

	params.model_regularize = MODEL_REGULARIZE;

	params.layers = LAYERS;

	params.optimizer = OPTIMIZE_ADAM;

	params.print_progress = PRINT_PROGRESS;
	params.print_progress_iterations = PRINT_EVERY_X_ITERATIONS;
	params.print_progress_sample_output = PRINT_SAMPLE_OUTPUT;

	srand( time ( NULL ) );

	if ( argc < 2 ) {
		printf("Usage: %s datafile [-r name_of_net_to_load]\n", argv[0]);
		return -1;
	}

	initialize_set(&set);

	fp = fopen(argv[1], "r");
	if ( fp == NULL ) {
		printf("Could not open file: %s\n", argv[1]);
		return -1;
	}

	while ( ( c = fgetc(fp) ) != EOF ) {
		set_insert_symbol(&set, (char)c );
		++file_size;
	}
	set_insert_symbol(&set, '.');
	fclose(fp);

	X_train = calloc(file_size+1, sizeof(int));
	if ( X_train == NULL )
		return -1;

	X_train[file_size] = (int) set_char_to_indx(&set,'.');

	Y_train = &X_train[1];

	fp = fopen(argv[1], "r");
	while ( ( c = fgetc(fp) ) != EOF ) 
		X_train[sz++] = set_char_to_indx(&set,c);
	fclose(fp);

	int layers = LAYERS;

	params.layers = layers;

	model_layers = calloc(layers, sizeof(lstm_model_t*));

	if ( model_layers == NULL ) {
		printf("Error in init!\n");
		exit(-1);
	}

	p = 0;

	while ( p < layers ) {
		lstm_init_model(set_get_features(&set), NEURONS, &model_layers[p], YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE, &params);	
		++p;
	}

	if ( argc >= 4 && !strcmp(argv[2], "-r") ) {
		lstm_read_net_layers(model_layers, argv[3]);
	}


	if ( argc >= 6 && !strcmp(argv[4], "-c") ) {
		do {
			clean = strchr(argv[5], '_');
			if ( clean != NULL )
				*clean = ' ';
		} while ( clean != NULL );

		lstm_output_string_from_string_layers(model_layers, &set, argv[5], 128);

	} else {

		printf("LSTM Neural net compiled: %s %s, %d Layers, Neurons: %d, Backprop Through Time: %d, LR: %lf, Mo: %lf, LA: %lf, LR-decrease: %lf\n",__DATE__, __TIME__, layers, NEURONS, MINI_BATCH_SIZE, params.learning_rate, params.momentum, params.lambda, params.learning_rate_decrease);

		signal(SIGINT, store_the_net_layers);

		lstm_train(
			model_layers[0],
			model_layers,
			&set,
			file_size,
			X_train,
			Y_train,
			ITERATIONS,
			layers
		);

	}


	free(model_layers);
	free(X_train);



	return 0;

}
