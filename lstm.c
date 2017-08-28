#include "lstm.h"

//					 Features,   Neurons,  &lstm model, 		zeros
int lstm_init_model(int F, int N, lstm_model_t** model_to_be_set, int zeros)
{
	int S = F + N;
	lstm_model_t* lstm = calloc(1, sizeof(lstm_model_t));
	if ( lstm == NULL )
		exit(-1);

	lstm->F = F;
	lstm->N = N;
	lstm->S = S;

	lstm->learning_rate = STD_LEARNING_RATE;

	if ( zeros ) {
		lstm->Wf = get_zero_matrix(N, S);
		lstm->Wi = get_zero_matrix(N, S);
		lstm->Wc = get_zero_matrix(N, S);
		lstm->Wo = get_zero_matrix(N, S);
		lstm->Wy = get_zero_matrix(F, N);
	} else {
		lstm->Wf = get_random_matrix(N, S);
		lstm->Wi = get_random_matrix(N, S);
		lstm->Wc = get_random_matrix(N, S);
		lstm->Wo = get_random_matrix(N, S);
		lstm->Wy = get_random_matrix(F, N);
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

	*model_to_be_set = lstm;

	return 0;
}
//					 lstm model to be freed
void lstm_free_model(lstm_model_t* lstm)
{
	free_matrix(lstm->Wf,lstm->N);
	free_matrix(lstm->Wi,lstm->N);
	free_matrix(lstm->Wc,lstm->N);
	free_matrix(lstm->Wo,lstm->N);
	free_matrix(lstm->Wy,lstm->F);

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

void sum_gradients(lstm_model_t* gradients, lstm_model_t* gradients_entry)
{
	matrix_add(gradients->Wy, gradients_entry->Wy, gradients->F, gradients->N);
	matrix_add(gradients->Wi, gradients_entry->Wi, gradients->N, gradients->S);
	matrix_add(gradients->Wc, gradients_entry->Wc, gradients->N, gradients->S);
	matrix_add(gradients->Wo, gradients_entry->Wo, gradients->N, gradients->S);
	matrix_add(gradients->Wf, gradients_entry->Wf, gradients->N, gradients->S);

	vectors_add(gradients->by, gradients_entry->by, gradients->F);
	vectors_add(gradients->bi, gradients_entry->bi, gradients->N);
	vectors_add(gradients->bc, gradients_entry->bc, gradients->N);
	vectors_add(gradients->bf, gradients_entry->bf, gradients->N);
	vectors_add(gradients->bo, gradients_entry->bo, gradients->N);
}

// A = A - alpha * dl/dA
void gradients_decend(lstm_model_t* model, lstm_model_t* gradients) {
	matrix_scalar_multiply(gradients->Wy, model->learning_rate, model->F, model->N);
	matrix_scalar_multiply(gradients->Wi, model->learning_rate, model->N, model->S);
	matrix_scalar_multiply(gradients->Wc, model->learning_rate, model->N, model->S);
	matrix_scalar_multiply(gradients->Wo, model->learning_rate, model->N, model->S);
	matrix_scalar_multiply(gradients->Wf, model->learning_rate, model->N, model->S);

	matrix_substract(model->Wy, gradients->Wy, model->F, model->N);
	matrix_substract(model->Wi, gradients->Wi, model->N, model->S);
	matrix_substract(model->Wc, gradients->Wc, model->N, model->S);
	matrix_substract(model->Wo, gradients->Wo, model->N, model->S);
	matrix_substract(model->Wf, gradients->Wf, model->N, model->S);

	vectors_mutliply_scalar(gradients->by, model->learning_rate, model->F);
	vectors_mutliply_scalar(gradients->bi, model->learning_rate, model->N);
	vectors_mutliply_scalar(gradients->bc, model->learning_rate, model->N);
	vectors_mutliply_scalar(gradients->bo, model->learning_rate, model->N);
	vectors_mutliply_scalar(gradients->bf, model->learning_rate, model->N);

	vectors_substract(model->by, gradients->by, model->F);
	vectors_substract(model->bi, gradients->bi, model->N);
	vectors_substract(model->bc, gradients->bc, model->N);
	vectors_substract(model->bf, gradients->bf, model->N);
	vectors_substract(model->bo, gradients->bo, model->N);
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
	matrix_set_to_zero(model->Wy, model->F, model->N);
	matrix_set_to_zero(model->Wi, model->N, model->S);
	matrix_set_to_zero(model->Wc, model->N, model->S);
	matrix_set_to_zero(model->Wo, model->N, model->S);
	matrix_set_to_zero(model->Wf, model->N, model->S);

	vector_set_to_zero(model->by, model->F);
	vector_set_to_zero(model->bi, model->N);
	vector_set_to_zero(model->bc, model->N);
	vector_set_to_zero(model->bf, model->N);
	vector_set_to_zero(model->bo, model->N);
}

//						model, number of training points, X_train, Y_train, number of iterations
void lstm_train_the_next(lstm_model_t* model, set_T* char_index_mapping, unsigned int training_points, int* X_train, int* Y_train, unsigned long iterations)
{
	int N,F,S;
	unsigned int i = 0;
	unsigned long n = 0;
	double loss = 0.0;
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

	while ( n < iterations ){
		i = 0;
		loss = 0.0;
		while ( i < training_points + 1){
			caches[i] = lstm_cache_container_init(N, F);
			++i;
		}

		i = 0;

		while ( i < training_points ) {
			lstm_forward_propagate(model, X_train[i], caches[i], caches[i+1]);
			loss += cross_entropy( caches[i+1]->probs, Y_train[i]);
			++i;
		}

		loss /= training_points; 

		lstm_init_model(F, N, &gradients, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE);

		i = training_points;

		lstm_values_next_cache_init(&d_next, N);
		
		lstm_init_model(F, N, &gradients_entry, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE);

		while ( i > 0 ) {

			lstm_zero_the_model(gradients_entry);

			lstm_backward_propagate(model, caches[i]->probs, Y_train[i-1], d_next, caches[i], gradients_entry, d_next);

			sum_gradients(gradients, gradients_entry);

			--i;
		}		

		lstm_free_model(gradients_entry);

		i = 0;
		while ( i < training_points + 1) {
			lstm_cache_container_free(caches[i]);
			free(caches[i]);
			++i;
		}

		lstm_values_next_cache_free(d_next);

		gradients_decend(model, gradients);
		lstm_free_model(gradients);

		printf("Iteration: %lu, Loss: %lf\n", n+1, loss);
		printf("===================\n");

		i = 0;
		while ( i < 50 + 1){
			caches[i] = lstm_cache_container_init( N, F);
			++i;
		}

		i = 0;
		char input = X_train[0];
		while ( i < 50 ) {
			lstm_forward_propagate(model, input, caches[i], caches[i+1]);
			input = set_probability_choice(char_index_mapping, caches[i+1]->probs);
			printf ( "%c", input );
			++i;
		}

		printf("\n===================\n");

		i = 0;
		while ( i < 50 + 1) {
			lstm_cache_container_free(caches[i]);
			free(caches[i]);
			++i;
		}
		++n;
	}

	free(tmp);

}










