#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

#include "lstm.h"
#include "set.h"
#include "layers.h"
#include "utilities.h"

#include "std_conf.h"

#define ITERATIONS 	100000000

lstm_model_t *model = NULL, *layer1 = NULL, *layer2 = NULL;
lstm_model_t **model_layers;
set_T set;

void store_the_net_layers(int signo)
{
	if ( model_layers != NULL ){
		lstm_store_net_layers(model_layers, STD_LOADABLE_NET_NAME, LAYERS);
		lstm_store_net_layers_as_json(model_layers, STD_JSON_NET_NAME, JSON_KEY_NAME_SET, &set, LAYERS);
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

void usage(char *argv[]) {
	printf("Usage: %s datafile [flag value]*\n", argv[0]);
	printf("\n");
	printf("Flags can be used to change the training procedure.\n");
	printf("The flags require a value to be passed as the following argument.\n");
	printf("    E.g., this is how you traing with a learning rate set to 0.03:\n");
	printf("        %s datafile -lr 0.03\n", argv[0]);
	printf("\n");
	printf("The following flags are available:\n");
	printf("    -r : read a previously trained network, the name of which is currently configured to be '%s'.\n", STD_LOADABLE_NET_NAME);
	printf("    -lr: learning rate that is to be used during training, see the example above.\n");
	printf("    -it: the number of iterations used for training (not to be confused with epochs).\n");
	printf("    -mb: mini batch size.\n");
	printf("    -dl: decrease the learning rate over time, according to lr(n+1) <- lr(n) / (1 + n/value).\n");
	printf("    -st: number of iterations between how the network is continously stored during training (.json and .net).\n");
	printf("\n");
	printf("Check std_conf.h to see what default values are used, these are set during compilation.\n");
	printf("\n");
	printf("%s compiled %s %s\n", argv[0], __DATE__, __TIME__);
	exit(1);
}

void parse_input_args(int argc, char** argv, lstm_model_parameters_t* params)
{
	int a = 0;

	while ( a < argc ) {

		if ( argc <= (a+1) )
			break; // All flags have values attributed to them

		if ( !strcmp(argv[a], "-r") ) {
			lstm_read_net_layers(model_layers, argv[a + 1], LAYERS);
		} else if ( !strcmp(argv[a], "-lr") ) {
			params->learning_rate = atof(argv[a + 1]);
			if ( params->learning_rate == 0.0 ) {
				usage(argv);
			}
		} else if ( !strcmp(argv[a], "-mb") ) {
			params->mini_batch_size = atoi(argv[a + 1]);
			if ( params->mini_batch_size <= 0 ) {
				usage(argv);
			}
		} else if ( !strcmp(argv[a], "-it") ) {
			params->iterations = (unsigned long) atol(argv[a + 1]);
			if ( params->iterations == 0 ) {
				usage(argv);
			}
		} else if ( !strcmp(argv[a], "-dl") ) {
			params->learning_rate_decrease = atof(argv[a + 1]);
			if ( params->learning_rate_decrease == 0 ) {
				usage(argv);
			}
			params->decrease_lr = 1;
		} else if ( !strcmp(argv[a], "-st") ) {
			params->store_network_every = atoi(argv[a + 1]);
			if ( params->store_network_every == 0 ) {
				usage(argv);
			}
		}

		a += 2;
	}
}

int main(int argc, char *argv[])
{
	int i = 0, c, p = 0;
	size_t file_size = 0, sz = 0;
	int *X_train, *Y_train;
	char * clean;
	FILE * fp;

	int layers = LAYERS;

	lstm_model_parameters_t params;
	memset(&params, 0, sizeof(params));

	params.iterations = ITERATIONS;
	params.loss_moving_avg = LOSS_MOVING_AVG;
	params.learning_rate = STD_LEARNING_RATE;
	params.momentum = STD_MOMENTUM;
	params.lambda = STD_LAMBDA;
	params.softmax_temp = SOFTMAX_TEMP;
	params.mini_batch_size = MINI_BATCH_SIZE;
	params.gradient_clip_limit = GRADIENT_CLIP_LIMIT;
	params.learning_rate_decrease = STD_LEARNING_RATE_DECREASE;
	params.stateful = STATEFUL;
	params.beta1 = 0.9;
	params.beta2 = 0.999;
	params.gradient_fit = GRADIENTS_FIT;
	params.gradient_clip = GRADIENTS_CLIP;
	params.decrease_lr = DECREASE_LR;
	params.model_regularize = MODEL_REGULARIZE;
	params.layers = LAYERS;
	params.optimizer = OPTIMIZE_ADAM;
	// Interaction configuration with the training of the network
	params.print_progress = PRINT_PROGRESS;
	params.print_progress_iterations = PRINT_EVERY_X_ITERATIONS;
	params.print_progress_sample_output = PRINT_SAMPLE_OUTPUT;
	params.print_progress_to_file = PRINT_SAMPLE_OUTPUT_TO_FILE;
	params.print_progress_number_of_chars = NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING;
	params.print_sample_output_to_file_arg = PRINT_SAMPLE_OUTPUT_TO_FILE_ARG;
	params.print_sample_output_to_file_name = PRINT_SAMPLE_OUTPUT_TO_FILE_NAME;
	params.store_progress_every_x_iterations = STORE_PROGRESS_EVERY_X_ITERATIONS;
	params.store_progress_file_name = PROGRESS_FILE_NAME;
	params.store_network_name_raw = STD_LOADABLE_NET_NAME;
	params.store_network_name_json = STD_LOADABLE_NET_NAME;
	params.store_char_indx_map_name = JSON_KEY_NAME_SET;

	srand( time ( NULL ) );

	if ( argc < 2 ) {
		usage(argv);
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

	model_layers = calloc(layers, sizeof(lstm_model_t*));

	if ( model_layers == NULL ) {
		printf("Error in init!\n");
		exit(-1);
	}

	p = 0;

	while ( p < layers ) {
		// All layers have the same training parameters
		lstm_init_model(set_get_features(&set), NEURONS, &model_layers[p], 0, &params);	
		++p;
	}

	parse_input_args(argc, argv, &params);

	if ( argc >= 6 && !strcmp(argv[4], "-c") ) {
		do {
			clean = strchr(argv[5], '_');
			if ( clean != NULL )
				*clean = ' ';
		} while ( clean != NULL );

		lstm_output_string_from_string_layers(model_layers, &set, argv[5], LAYERS, 128);

	} else {

		printf("LSTM Neural net compiled: %s %s, %d Layers, Neurons: %d, Backprop Through Time: %d, LR: %lf, Mo: %lf, LA: %lf, LR-decrease: %lf\n",__DATE__, __TIME__, layers, NEURONS, MINI_BATCH_SIZE, params.learning_rate, params.momentum, params.lambda, params.learning_rate_decrease);

		signal(SIGINT, store_the_net_layers);

		lstm_train(
			model_layers,
			&params,
			&set,
			file_size,
			X_train,
			Y_train,
			layers
		);

	}


	free(model_layers);
	free(X_train);



	return 0;

}
