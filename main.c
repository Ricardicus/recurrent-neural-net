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

void store_the_net(int signo)
{

#ifdef ONE_LAYER

	if ( model != NULL ){
		lstm_store_net(model, STD_LOADABLE_NET_NAME);
		printf("\nStored the net\n");	
	} else {
		printf("Failed to store the net!");
		exit(-1);
	}

	exit(0);
	return;
#endif

#ifdef TWO_LAYERS

	if ( layer1 != NULL && layer2 != NULL ){
		lstm_store_net_two_layers(layer1, layer2, STD_LOADABLE_NET_NAME);
		printf("\nStored the net\n");
	} else {
		printf("Failed to store the net!");
		exit(-1);
	}

	exit(0);
	return;
#endif

}

int main(int argc, char *argv[])
{
	int i = 0, c;
	size_t file_size = 0, sz = 0;
	int *X_train, *Y_train;
	char * clean;
	FILE * fp;
	set_T set;

	lstm_model_parameters_t params;

	params.loss_moving_avg = LOSS_MOVING_AVG;
	params.learning_rate = STD_LEARNING_RATE;
	params.momentum = STD_MOMENTUM;
	params.lambda = STD_LAMBDA;
	params.softmax_temp = SOFTMAX_TEMP;
	params.mini_batch_size = MINI_BATCH_SIZE;
	params.gradient_clip_limit = GRADIENT_CLIP_LIMIT;
	params.learning_rate_decrease = STD_LEARNING_RATE_DECREASE;
	params.learning_rate_decrease_threshold = STD_LEARNING_RATE_THRESHOLD;

	params.beta1 = 0.9;
	params.beta2 = 0.999;

	srand( time ( NULL ) );

	if ( argc < 2 ) {
		printf("Usage: ./binary datafile [-r name_of_net_to_load]\n");
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

#ifdef ONE_LAYER

	lstm_init_model(set_get_features(&set), NEURONS, &model, YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE, &params);

	if ( argc >= 4 && !strcmp(argv[2], "-r") ) {
		lstm_read_net(model, argv[3]);
	}

	if ( argc >= 6 && !strcmp(argv[4], "-c") ) {
		do {
			clean = strchr(argv[5], '_');
			if ( clean != NULL )
				*clean = ' ';
		} while ( clean != NULL );
		lstm_output_string_from_string(model, &set, argv[5], 100);
	} else {
	#ifdef DECREASE_LR
		printf("LSTM Neural net compiled: %s %s, Neurons: %d, LR: %lf, Mo: %lf, LA: %lf, LR-decrease: %lf\n", __DATE__, __TIME__, NEURONS, params.learning_rate, params.momentum, params.lambda, params.learning_rate_decrease);
	#else
		printf("LSTM Neural net compiled: %s %s, Neurons: %d, LR: %lf, Mo: %lf, LA: %lf\n", __DATE__, __TIME__, NEURONS, params.learning_rate, params.momentum, params.lambda);
	#endif
		signal(SIGINT, store_the_net);
		lstm_train_the_net(model, &set, file_size, X_train, Y_train, ITERATIONS);
	}

	free(X_train);

	return 0;
#endif 

#ifdef TWO_LAYERS


	lstm_init_model(set_get_features(&set), NEURONS, &layer1, YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE, &params);
	lstm_init_model(set_get_features(&set), NEURONS, &layer2, YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE, &params);

	if ( argc >= 4 && !strcmp(argv[2], "-r") ) {
		lstm_read_net_two_layers(layer1, layer2, argv[3]);
	}

	if ( argc >= 6 && !strcmp(argv[4], "-c") ) {
		do {
			clean = strchr(argv[5], '_');
			if ( clean != NULL )
				*clean = ' ';
		} while ( clean != NULL );

		lstm_output_string_from_string_two_layers(layer1, layer2, &set, argv[5], 100);
	} else {
	#ifdef DECREASE_LR
		printf("LSTM Neural net compiled: %s %s, Two layers, Neurons: %d, LR: %lf, Mo: %lf, LA: %lf, LR-decrease: %lf\n",__DATE__, __TIME__, NEURONS, params.learning_rate, params.momentum, params.lambda, params.learning_rate_decrease);
	#else
		printf("LSTM Neural net compiled: %s %s, Two layers, Neurons: %d, LR: %lf, Mo: %lf, LA: %lf\n", __DATE__, __TIME__, NEURONS, params.learning_rate, params.momentum, params.lambda);
	#endif

		signal(SIGINT, store_the_net);

		lstm_train_the_net_two_layers(layer1, layer1, layer2, &set, file_size, X_train, Y_train, ITERATIONS);
	}

	free(X_train);

	return 0;

#endif

}
