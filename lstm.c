#include "lstm.h"

int debug_print_on = 1;

//					 Features,   Neurons,  &lstm model, 		zeros
int lstm_init_model(int F, int N, lstm_model_t** model_to_be_set, int zeros, lstm_model_parameters_t * params)
{
	int S = F + N;
	lstm_model_t* lstm = calloc(1, sizeof(lstm_model_t));
	if ( lstm == NULL )
		exit(-1);

	lstm->F = F;
	lstm->N = N;
	lstm->S = S;

	lstm->params = params;

	if ( zeros ) {
		lstm->Wf = get_zero_vector(N * S);
		lstm->Wi = get_zero_vector(N * S);
		lstm->Wc = get_zero_vector(N * S);
		lstm->Wo = get_zero_vector(N * S);
		lstm->Wy = get_zero_vector(F * N);
	} else {
		lstm->Wf = get_random_vector(N * S, N);
		lstm->Wi = get_random_vector(N * S, N);
		lstm->Wc = get_random_vector(N * S, N);
		lstm->Wo = get_random_vector(N * S, N);
		lstm->Wy = get_random_vector(F * N, F);
	}

	lstm->bf = get_zero_vector(N);
	lstm->bi = get_zero_vector(N);
	lstm->bc = get_zero_vector(N);
	lstm->bo = get_zero_vector(N);
	lstm->by = get_zero_vector(F);

	lstm->dldhf = get_zero_vector(N);
	lstm->dldhi = get_zero_vector(N);
	lstm->dldhc = get_zero_vector(N);
	lstm->dldho = get_zero_vector(N);
	lstm->dldc  = get_zero_vector(N);
	lstm->dldh  = get_zero_vector(N);

	lstm->dldXc = get_zero_vector(S);
	lstm->dldXo = get_zero_vector(S);
	lstm->dldXi = get_zero_vector(S);
	lstm->dldXf = get_zero_vector(S);

	// Gradient descent momentum caches
	lstm->Wfm = get_zero_vector(N * S);
	lstm->Wim = get_zero_vector(N * S);
	lstm->Wcm = get_zero_vector(N * S);
	lstm->Wom = get_zero_vector(N * S);
	lstm->Wym = get_zero_vector(F * N);

	lstm->bfm = get_zero_vector(N);
	lstm->bim = get_zero_vector(N);
	lstm->bcm = get_zero_vector(N);
	lstm->bom = get_zero_vector(N);
	lstm->bym = get_zero_vector(F);

	*model_to_be_set = lstm;

	return 0;
}
//					 lstm model to be freed
void lstm_free_model(lstm_model_t* lstm)
{
	free_vector(&lstm->Wf);
	free_vector(&lstm->Wi);
	free_vector(&lstm->Wc);
	free_vector(&lstm->Wo);
	free_vector(&lstm->Wy);

	free_vector(&lstm->bf);
	free_vector(&lstm->bi);
	free_vector(&lstm->bc);
	free_vector(&lstm->bo);
	free_vector(&lstm->by);

	free_vector(&lstm->dldhf);
	free_vector(&lstm->dldhi);
	free_vector(&lstm->dldhc);
	free_vector(&lstm->dldho);
	free_vector(&lstm->dldc);
	free_vector(&lstm->dldh);

	free_vector(&lstm->dldXc);
	free_vector(&lstm->dldXo);
	free_vector(&lstm->dldXi);
	free_vector(&lstm->dldXf);

	free_vector(&lstm->Wfm);
	free_vector(&lstm->Wim);
	free_vector(&lstm->Wcm);
	free_vector(&lstm->Wom);
	free_vector(&lstm->Wym);

	free_vector(&lstm->bfm);
	free_vector(&lstm->bim);
	free_vector(&lstm->bcm);
	free_vector(&lstm->bom);
	free_vector(&lstm->bym);

	free(lstm);
}

void lstm_cache_container_free(lstm_values_cache_t* cache_to_be_freed)
{
	free_vector(&(cache_to_be_freed)->probs);
	free_vector(&(cache_to_be_freed)->c);
	free_vector(&(cache_to_be_freed)->h);
	free_vector(&(cache_to_be_freed)->c_old);
	free_vector(&(cache_to_be_freed)->h_old);
	free_vector(&(cache_to_be_freed)->X);
	free_vector(&(cache_to_be_freed)->hf);
	free_vector(&(cache_to_be_freed)->hi);
	free_vector(&(cache_to_be_freed)->ho);
	free_vector(&(cache_to_be_freed)->hc);
	free_vector(&(cache_to_be_freed)->tanh_c_cache);
}

lstm_values_cache_t*  lstm_cache_container_init(int N, int F)
{
	int S = N + F;

	lstm_values_cache_t* cache = calloc(1, sizeof(lstm_values_cache_t));

	if ( cache == NULL )
		exit(-1);

	cache->probs = get_zero_vector(F);
	cache->c = get_zero_vector(N);
	cache->h = get_zero_vector(N);
	cache->c_old = get_zero_vector(N);
	cache->h_old = get_zero_vector(N);
	cache->X = get_zero_vector(S);
	cache->hf = get_zero_vector(N);
	cache->hi = get_zero_vector(N);
	cache->ho = get_zero_vector(N);
	cache->hc = get_zero_vector(N);
	cache->tanh_c_cache = get_zero_vector(N);

	return cache;
}

int gradients_fit(lstm_model_t* gradients, double limit)
{
	int msg = 0;
	msg += vectors_fit(gradients->Wy, limit, gradients->F * gradients->N);
	msg += vectors_fit(gradients->Wi, limit, gradients->N * gradients->S);
	msg += vectors_fit(gradients->Wc, limit, gradients->N * gradients->S);
	msg += vectors_fit(gradients->Wo, limit, gradients->N * gradients->S);
	msg += vectors_fit(gradients->Wf, limit, gradients->N * gradients->S);

	msg += vectors_fit(gradients->by, limit, gradients->F);
	msg += vectors_fit(gradients->bi, limit, gradients->N);
	msg += vectors_fit(gradients->bc, limit, gradients->N);
	msg += vectors_fit(gradients->bf, limit, gradients->N);
	msg += vectors_fit(gradients->bo, limit, gradients->N);

	return msg;
}

int gradients_clip(lstm_model_t* gradients, double limit)
{
	int msg = 0;
	msg += vectors_clip(gradients->Wy, limit, gradients->F * gradients->N);
	msg += vectors_clip(gradients->Wi, limit, gradients->N * gradients->S);
	msg += vectors_clip(gradients->Wc, limit, gradients->N * gradients->S);
	msg += vectors_clip(gradients->Wo, limit, gradients->N * gradients->S);
	msg += vectors_clip(gradients->Wf, limit, gradients->N * gradients->S);

	msg += vectors_clip(gradients->by, limit, gradients->F);
	msg += vectors_clip(gradients->bi, limit, gradients->N);
	msg += vectors_clip(gradients->bc, limit, gradients->N);
	msg += vectors_clip(gradients->bf, limit, gradients->N);
	msg += vectors_clip(gradients->bo, limit, gradients->N);

	return msg;
}

void sum_gradients(lstm_model_t* gradients, lstm_model_t* gradients_entry)
{
	vectors_add(gradients->Wy, gradients_entry->Wy, gradients->F * gradients->N);
	vectors_add(gradients->Wi, gradients_entry->Wi, gradients->N * gradients->S);
	vectors_add(gradients->Wc, gradients_entry->Wc, gradients->N * gradients->S);
	vectors_add(gradients->Wo, gradients_entry->Wo, gradients->N * gradients->S);
	vectors_add(gradients->Wf, gradients_entry->Wf, gradients->N * gradients->S);

	vectors_add(gradients->by, gradients_entry->by, gradients->F);
	vectors_add(gradients->bi, gradients_entry->bi, gradients->N);
	vectors_add(gradients->bc, gradients_entry->bc, gradients->N);
	vectors_add(gradients->bf, gradients_entry->bf, gradients->N);
	vectors_add(gradients->bo, gradients_entry->bo, gradients->N);
}

// A -= alpha * Am_hat / (np.sqrt(Rm_hat) + epsilon)
// Am_hat = Am / ( 1 - betaM ^ (iteration) )
// Rm_hat = Rm / ( 1 - betaR ^ (iteration) )

void gradients_adam_optimizer(lstm_model_t* model, lstm_model_t* gradients) 
{


}

// A = A - alpha * m, m = momentum * m + ( 1 - momentum ) * dldA
void gradients_decend(lstm_model_t* model, lstm_model_t* gradients) {
	
	// Computing momumentum * m
	vectors_mutliply_scalar(gradients->Wym, model->params->momentum, model->F * model->N);
	vectors_mutliply_scalar(gradients->Wim, model->params->momentum, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wcm, model->params->momentum, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wom, model->params->momentum, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wfm, model->params->momentum, model->N * model->S);

	vectors_mutliply_scalar(gradients->bym, model->params->momentum, model->F);
	vectors_mutliply_scalar(gradients->bim, model->params->momentum, model->N);
	vectors_mutliply_scalar(gradients->bcm, model->params->momentum, model->N);
	vectors_mutliply_scalar(gradients->bom, model->params->momentum, model->N);
	vectors_mutliply_scalar(gradients->bfm, model->params->momentum, model->N);

	// Computing m = momentum * m + (1 - momentum) * dldA
	vectors_add_scalar_multiply(gradients->Wym, gradients->Wy, model->F * model->N, 1.0 - model->params->momentum);
	vectors_add_scalar_multiply(gradients->Wim, gradients->Wi, model->N * model->S, 1.0 - model->params->momentum);
	vectors_add_scalar_multiply(gradients->Wcm, gradients->Wc, model->N * model->S, 1.0 - model->params->momentum);
	vectors_add_scalar_multiply(gradients->Wom, gradients->Wo, model->N * model->S, 1.0 - model->params->momentum);
	vectors_add_scalar_multiply(gradients->Wfm, gradients->Wf, model->N * model->S, 1.0 - model->params->momentum);

	vectors_add_scalar_multiply(gradients->bym, gradients->by, model->F, 1.0 - model->params->momentum);
	vectors_add_scalar_multiply(gradients->bim, gradients->bi, model->N, 1.0 - model->params->momentum);
	vectors_add_scalar_multiply(gradients->bcm, gradients->bc, model->N, 1.0 - model->params->momentum);
	vectors_add_scalar_multiply(gradients->bom, gradients->bo, model->N, 1.0 - model->params->momentum);
	vectors_add_scalar_multiply(gradients->bfm, gradients->bf, model->N, 1.0 - model->params->momentum);

	// Computing A = A - alpha * m
	vectors_substract_scalar_multiply(model->Wy, gradients->Wym, model->F * model->N, model->params->learning_rate);
	vectors_substract_scalar_multiply(model->Wi, gradients->Wim, model->N * model->S, model->params->learning_rate);
	vectors_substract_scalar_multiply(model->Wc, gradients->Wcm, model->N * model->S, model->params->learning_rate);
	vectors_substract_scalar_multiply(model->Wo, gradients->Wom, model->N * model->S, model->params->learning_rate);
	vectors_substract_scalar_multiply(model->Wf, gradients->Wfm, model->N * model->S, model->params->learning_rate);

	vectors_substract_scalar_multiply(model->by, gradients->bym, model->F, model->params->learning_rate);
	vectors_substract_scalar_multiply(model->bi, gradients->bim, model->N, model->params->learning_rate);
	vectors_substract_scalar_multiply(model->bc, gradients->bcm, model->N, model->params->learning_rate);
	vectors_substract_scalar_multiply(model->bf, gradients->bfm, model->N, model->params->learning_rate);
	vectors_substract_scalar_multiply(model->bo, gradients->bom, model->N, model->params->learning_rate);
}

void lstm_values_next_cache_init(lstm_values_next_cache_t** d_next_to_set, int N)
{
	lstm_values_next_cache_t * d_next = calloc(1, sizeof(lstm_values_next_cache_t));
	if ( d_next == NULL )
		return;
	init_zero_vector(&d_next->dldh_next, N);
	init_zero_vector(&d_next->dldc_next, N);
	*d_next_to_set = d_next;
}
void lstm_values_next_cache_free(lstm_values_next_cache_t* d_next)
{
	free_vector(&d_next->dldc_next);
	free_vector(&d_next->dldh_next);
	free(d_next);
}

//							model, input,  state and cache values, &probs, &state and cache values
void lstm_forward_propagate(lstm_model_t* model, int X_index, lstm_values_cache_t* cache_in, lstm_values_cache_t* cache_out)
{
	int N, F, S, i = 0;
	double *h_old, *c_old, *X_one_hot;

	h_old = cache_in->h;
	c_old = cache_in->c;

	N = model->N;
	F = model->F;
	S = model->S;

	double tmp[N]; // VLA must be supported.. May cause portability problems.. If so use init_zeros_vector (will be slower).

	copy_vector(cache_out->h_old, h_old, N);
	copy_vector(cache_out->c_old, c_old, N);

	X_one_hot = cache_out->X;

	while ( i < S ) {
		if ( i < N ) {
			X_one_hot[i] = h_old[i];
		} else {
			X_one_hot[i] = 0.0;
		}
		++i;
	}

	X_one_hot[N + X_index] = 1.0;

	// Fully connected + sigmoid layers 
	fully_connected_forward(cache_out->hf, model->Wf, X_one_hot, model->bf, N, S);
	sigmoid_forward(cache_out->hf, cache_out->hf, N);
	
	fully_connected_forward(cache_out->hi, model->Wi, X_one_hot, model->bi, N, S);
	sigmoid_forward(cache_out->hi, cache_out->hi, N);
	
	fully_connected_forward(cache_out->ho, model->Wo, X_one_hot, model->bo, N, S);
	sigmoid_forward(cache_out->ho, cache_out->ho, N);

	fully_connected_forward(cache_out->hc, model->Wc, X_one_hot, model->bc, N, S);
	sigmoid_forward(cache_out->hc, cache_out->hc, N);

	// c = hf * c_old + hi * hc
	copy_vector(cache_out->c, cache_out->hf, N);
	vectors_multiply(cache_out->c, c_old, N);
	copy_vector(tmp, cache_out->hi, N);
	vectors_multiply(tmp, cache_out->hc, N);

	vectors_add(cache_out->c, tmp, N);

	// h = ho * tanh_c_cache
	tanh_forward(cache_out->tanh_c_cache, cache_out->c, N);
	copy_vector(cache_out->h, cache_out->ho, N);
	vectors_multiply(cache_out->h, cache_out->tanh_c_cache, N);
	
	// probs = softmax ( Wy*h + by )
	fully_connected_forward(cache_out->probs, model->Wy, cache_out->h, model->by, F, N);
	softmax_layers_forward(cache_out->probs, cache_out->probs, F);

	copy_vector(cache_out->X, X_one_hot, S);

}
//							model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
void lstm_backward_propagate(lstm_model_t* model, double* y_probabilities, int y_correct, lstm_values_next_cache_t* d_next, lstm_values_cache_t* cache_in, lstm_model_t* gradients, lstm_values_next_cache_t* cache_out)
{
	double *h,*c,*dldh_next,*dldc_next, *dldy, *dldh, *dldho, *dldhf, *dldhi, *dldhc, *dldc;
	int N, F, S;

	N = model->N;
	F = model->F;
	S = model->S;

	// model cache
	dldh = model->dldh;
	dldc = model->dldc;
	dldho = model->dldho;
	dldhi = model->dldhi;
	dldhf = model->dldhf;
	dldhc = model->dldhc;

	h = cache_in->h;
	c = cache_in->c;

	dldh_next = d_next->dldh_next;
	dldc_next = d_next->dldc_next;

	dldy = y_probabilities;
	dldy[y_correct] -= 1.0;

	fully_connected_backward(dldy, model->Wy, h, gradients->Wy, dldh, gradients->by, F, N);
	vectors_add(dldh, dldh_next, N);

	copy_vector(dldho, dldh, N);
	vectors_multiply(dldho, cache_in->tanh_c_cache, N);
	sigmoid_backward(dldho, cache_in->ho, dldho, N);

	copy_vector(dldc, dldh, N);
	vectors_multiply(dldc, cache_in->ho, N);
	tanh_backward(dldc, cache_in->tanh_c_cache, dldc, N);
	vectors_add(dldc, dldc_next, N);

	copy_vector(dldhf, dldc, N);
	vectors_multiply(dldhf, cache_in->c_old, N);
	sigmoid_backward(dldhf, cache_in->hf, dldhf, N);

	copy_vector(dldhi, cache_in->hc, N);
	vectors_multiply(dldhi, dldc, N);
	sigmoid_backward(dldhi, cache_in->hi, dldhi, N);

	copy_vector(dldhc, cache_in->hi, N);
	vectors_multiply(dldhc, dldc, N);
	tanh_backward(dldhc, cache_in->hc, dldhc, N);

	fully_connected_backward(dldhi, model->Wi, cache_in->X, gradients->Wi, gradients->dldXi, gradients->bi, N, S);
	fully_connected_backward(dldhc, model->Wc, cache_in->X, gradients->Wc, gradients->dldXc, gradients->bc, N, S);
	fully_connected_backward(dldho, model->Wo, cache_in->X, gradients->Wo, gradients->dldXo, gradients->bo, N, S);
	fully_connected_backward(dldhf, model->Wf, cache_in->X, gradients->Wf, gradients->dldXf, gradients->bf, N, S);
	
	// dldXi will work as a temporary substitute for dldX (where we get extract dh_next from!)
	vectors_add(gradients->dldXi, gradients->dldXc, S);
	vectors_add(gradients->dldXi, gradients->dldXo, S);
	vectors_add(gradients->dldXi, gradients->dldXf, S);

	copy_vector(cache_out->dldh_next, gradients->dldXi, N);
	copy_vector(cache_out->dldc_next, cache_in->hf, N);
	vectors_multiply(cache_out->dldc_next, dldc, N);
}

void lstm_zero_the_model(lstm_model_t * model)
{
	vector_set_to_zero(model->Wy, model->F * model->N);
	vector_set_to_zero(model->Wi, model->N * model->S);
	vector_set_to_zero(model->Wc, model->N * model->S);
	vector_set_to_zero(model->Wo, model->N * model->S);
	vector_set_to_zero(model->Wf, model->N * model->S);

	vector_set_to_zero(model->by, model->F);
	vector_set_to_zero(model->bi, model->N);
	vector_set_to_zero(model->bc, model->N);
	vector_set_to_zero(model->bf, model->N);
	vector_set_to_zero(model->bo, model->N);
}

void lstm_zero_d_next(lstm_values_next_cache_t * d_next)
{
	vector_set_to_zero(d_next->dldh_next, NEURONS);
	vector_set_to_zero(d_next->dldc_next, NEURONS);
}

void lstm_cache_container_set_start(lstm_values_cache_t * cache)
{
	// State variables set to zero
	vector_set_to_zero(cache->h, NEURONS); 
	vector_set_to_zero(cache->c, NEURONS); 
}


void lstm_store_net(lstm_model_t* model, const char * filename) 
{
	FILE * fp;

	fp = fopen(filename, "w");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for writing.\n", filename);
		return;
	}

	vector_store(model->Wy, model->F * model->N, fp);
	vector_store(model->Wi, model->N * model->S, fp);
	vector_store(model->Wc, model->N * model->S, fp);
	vector_store(model->Wo, model->N * model->S, fp);
	vector_store(model->Wf, model->N * model->S, fp);

	vector_store(model->by, model->F, fp);
	vector_store(model->bi, model->N, fp);
	vector_store(model->bc, model->N, fp);
	vector_store(model->bf, model->N, fp);
	vector_store(model->bo, model->N, fp);

	fclose(fp);

}

void lstm_read_net(lstm_model_t* model, const char * filename) 
{
	FILE * fp;

	fp = fopen(filename, "r");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for writing.\n", filename);
		return;
	}

	vector_store(model->Wy, model->F * model->N, fp);
	vector_store(model->Wi, model->N * model->S, fp);
	vector_store(model->Wc, model->N * model->S, fp);
	vector_store(model->Wo, model->N * model->S, fp);
	vector_store(model->Wf, model->N * model->S, fp);

	vector_read(model->by, model->F, fp);
	vector_read(model->bi, model->N, fp);
	vector_read(model->bc, model->N, fp);
	vector_read(model->bf, model->N, fp);
	vector_read(model->bo, model->N, fp);

	printf("Loaded the net: %s\n", filename);
	fclose(fp);
}

void lstm_output_string(lstm_model_t *model, set_T* char_index_mapping, char in, int length) 
{
	lstm_values_cache_t * cache;
	int i = 0;
	char input = in;

	cache = lstm_cache_container_init(model->N, model->F);

	while ( i < length ) {
		lstm_forward_propagate(model, set_char_to_indx(char_index_mapping,input) , cache, cache);
		input = set_probability_choice(char_index_mapping, cache->probs);
		printf ( "%c", input );
		++i;
	}
}

void lstm_store_progress(unsigned int n, double loss)
{
	FILE * fp;

	fp = fopen(PROGRESS_FILE_NAME, "a");
	if ( fp != NULL ) {
		fprintf(fp, "%u,%lf\n",n,loss);
		fclose(fp);
	}

}

//						model, number of training points, X_train, Y_train, number of iterations
void lstm_train_the_next(lstm_model_t* model, set_T* char_index_mapping, unsigned int training_points, int* X_train, int* Y_train, unsigned long iterations)
{
	int N,F,S, status = 0;
	unsigned int i = 0, b = 0, q = 0, e1 = 0, e2 = 0, record_iteration = 0;
	unsigned long n = 0, decrease_threshold = model->params->learning_rate_decrease_threshold;
	double loss = -1, loss_tmp = 0.0, record_keeper = 0.0;
	lstm_values_cache_t **caches, **tmp; 
	lstm_values_next_cache_t *d_next = NULL;
	lstm_model_t *gradients, *gradients_entry = NULL;

	N = model->N;
	F = model->F;
	S = model->S;

	caches = calloc(training_points + 1, sizeof(lstm_values_cache_t*));
	if ( caches == NULL ) 
		return;

	tmp = caches;

	lstm_init_model(F, N, &gradients, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);
	lstm_values_next_cache_init(&d_next, N);	
	lstm_init_model(F, N, &gradients_entry, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);

	i = 0;
	while ( i < training_points + 1){
		caches[i] = lstm_cache_container_init(N, F);
		++i;
	}

	i = 0; b = 0;
	while ( n < iterations ){
		int bold = b;
		b = i;

		loss_tmp = 0.0;

		lstm_cache_container_set_start(caches[b]);

		q = 0;
		while ( q < model->params->mini_batch_size ) {
			e1 = i % training_points;
			e2 = ( e1 + 1 ) % training_points;
			lstm_forward_propagate(model, X_train[e1], caches[e1], caches[e2]);
			loss_tmp += cross_entropy( caches[e2]->probs, Y_train[e1]);
			++i; ++q;
		}

		loss_tmp /= (q+1); 

		if ( loss < 0 )
			loss = loss_tmp;

		loss = loss_tmp * model->params->loss_moving_avg + (1 - model->params->loss_moving_avg) * loss;

		if ( n == 0 ) 
			record_keeper = loss;

		if ( loss < record_keeper ){
			record_keeper = loss;
			
			/* if ( n - record_iteration > model->params->learning_rate_decrease_threshold ) {
				model->params->learning_rate *= model->params->learning_rate_decrease;
				printf("learning_rate: %lf\n", model->params->learning_rate);
			} */ 

			record_iteration = n;

		}

		lstm_zero_the_model(gradients);

		lstm_zero_d_next(d_next);

		q = model->params->mini_batch_size;
		while ( q > 0 ) {
			e1 = i % training_points;
			e2 = ( training_points + e1 - 1 ) % training_points;
			lstm_zero_the_model(gradients_entry);

			lstm_backward_propagate(model, caches[e1]->probs, Y_train[e2], d_next, caches[e1], gradients_entry, d_next);

			//gradients_fit(gradients_entry, model->params->gradient_clip_limit);

			sum_gradients(gradients, gradients_entry);
			i--; q--;
		}

		if ( gradients_clip(gradients, model->params->gradient_clip_limit) )
			status = 1;

		gradients_decend(model, gradients);

		if ( !( n % PRINT_EVERY_X_ITERATIONS ) ) {

			if (status) {
				printf("clipped the gradients this time..\n");
			}

			status = 0;
			printf("Iteration: %lu, Loss: %lf, record: %lf (iteration: %d)\n", n, loss, record_keeper, record_iteration);
			printf("===================\n");

			lstm_output_string(model, char_index_mapping, X_train[b], NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING);

			printf("\n===================\n");
			
			// Flushing stdout
			fflush(stdout);
		}

		if ( !(n % STORE_EVERY_X_ITERATIONS ) && n > 0 )
			lstm_store_net(model, STD_LOADABLE_NET_NAME);

		if ( !(n % STORE_PROGRESS_EVERY_X_ITERATIONS ))
			lstm_store_progress(n, loss);


		i = b + model->params->mini_batch_size;
		if ( i >= training_points )
			i = 0;

		//model->params->learning_rate = model->params->learning_rate / ( 1.0 + n / model->params->learning_rate_decrease );

		++n;
	}

	lstm_values_next_cache_free(d_next);

	i = 0;
	while ( i < training_points + 1) {
		lstm_cache_container_free(caches[i]);
		free(caches[i]);
		++i;
	}

	lstm_free_model(gradients_entry);
	lstm_free_model(gradients);

	free(tmp);

}








