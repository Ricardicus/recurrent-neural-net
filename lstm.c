#include "lstm.h"

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
		lstm->Wf = get_random_vector(N * S, S);
		lstm->Wi = get_random_vector(N * S, S);
		lstm->Wc = get_random_vector(N * S, S);
		lstm->Wo = get_random_vector(N * S, S);
		lstm->Wy = get_random_vector(F * N, N);
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
	free_vector(&(cache_to_be_freed)->probs_before_sigma);
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
	cache->probs_before_sigma = get_zero_vector(F);
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

void lstm_values_next_cache_init(lstm_values_next_cache_t** d_next_to_set, int N, int F)
{
	lstm_values_next_cache_t * d_next = calloc(1, sizeof(lstm_values_next_cache_t));
	if ( d_next == NULL )
		return;
	init_zero_vector(&d_next->dldh_next, N);
	init_zero_vector(&d_next->dldc_next, N);
	init_zero_vector(&d_next->dldY_pass, F);
	*d_next_to_set = d_next;
}
void lstm_values_next_cache_free(lstm_values_next_cache_t* d_next)
{
	free_vector(&d_next->dldc_next);
	free_vector(&d_next->dldh_next);
	free_vector(&d_next->dldY_pass);
	free(d_next);
}

//							model, input,  state and cache values, &probs, &state and cache values
void lstm_forward_propagate(lstm_model_t* model, double * input, lstm_values_cache_t* cache_in, lstm_values_cache_t* cache_out, int softmax)
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
			X_one_hot[i] = input[i - N];
		}
		++i;
	}

	// Fully connected + sigmoid layers 
	fully_connected_forward(cache_out->hf, model->Wf, X_one_hot, model->bf, N, S);
	sigmoid_forward(cache_out->hf, cache_out->hf, N);
	
	fully_connected_forward(cache_out->hi, model->Wi, X_one_hot, model->bi, N, S);
	sigmoid_forward(cache_out->hi, cache_out->hi, N);
	
	fully_connected_forward(cache_out->ho, model->Wo, X_one_hot, model->bo, N, S);
	sigmoid_forward(cache_out->ho, cache_out->ho, N);

	fully_connected_forward(cache_out->hc, model->Wc, X_one_hot, model->bc, N, S);
	tanh_forward(cache_out->hc, cache_out->hc, N);

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
	
	if  (softmax > 0 ){
		softmax_layers_forward(cache_out->probs, cache_out->probs, F);
	} else {
		copy_vector(cache_out->probs_before_sigma, cache_out->probs, F);
		sigmoid_forward(cache_out->probs, cache_out->probs_before_sigma, F);
	}

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

	if ( y_correct >= 0 ){
		dldy[y_correct] -= 1.0;
	} else {
		sigmoid_backward(y_probabilities, cache_in->probs_before_sigma, dldy, F);
	}

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

	// To pass on to next layer
	copy_vector(cache_out->dldY_pass, &gradients->dldXi[N], F);
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

	vector_set_to_zero(model->dldhf, model->N);
	vector_set_to_zero(model->dldhi, model->N);
	vector_set_to_zero(model->dldhc, model->N);
	vector_set_to_zero(model->dldho, model->N);
	vector_set_to_zero(model->dldc, model->N);
	vector_set_to_zero(model->dldh, model->N);

	vector_set_to_zero(model->dldXc, model->S);
	vector_set_to_zero(model->dldXo, model->S);
	vector_set_to_zero(model->dldXi, model->S);
	vector_set_to_zero(model->dldXf, model->S);
}

void lstm_zero_d_next(lstm_values_next_cache_t * d_next, int features)
{
	vector_set_to_zero(d_next->dldh_next, NEURONS);
	vector_set_to_zero(d_next->dldc_next, NEURONS);
	vector_set_to_zero(d_next->dldY_pass, features );
}

void lstm_cache_container_set_start(lstm_values_cache_t * cache)
{
	// State variables set to zero
	vector_set_to_zero(cache->h, NEURONS); 
	vector_set_to_zero(cache->c, NEURONS); 

	vector_set_to_zero(cache->c_old, NEURONS); 
	vector_set_to_zero(cache->h_old, NEURONS); 
	vector_set_to_zero(cache->X, NEURONS); 
	vector_set_to_zero(cache->hf, NEURONS); 
	vector_set_to_zero(cache->hi, NEURONS); 
	vector_set_to_zero(cache->ho, NEURONS); 
	vector_set_to_zero(cache->hc, NEURONS); 
	vector_set_to_zero(cache->tanh_c_cache, NEURONS); 
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

void lstm_store_net_two_layers(lstm_model_t* layer1, lstm_model_t* layer2, const char * filename) 
{
	// Will only work for ( layer1->N, layer1->F ) == ( layer2->N, layer2->F )
	FILE * fp;

	fp = fopen(filename, "w");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for writing.\n", filename);
		return;
	}

	vector_store(layer1->Wy, layer1->F * layer1->N, fp);
	vector_store(layer1->Wi, layer1->N * layer1->S, fp);
	vector_store(layer1->Wc, layer1->N * layer1->S, fp);
	vector_store(layer1->Wo, layer1->N * layer1->S, fp);
	vector_store(layer1->Wf, layer1->N * layer1->S, fp);

	vector_store(layer1->by, layer1->F, fp);
	vector_store(layer1->bi, layer1->N, fp);
	vector_store(layer1->bc, layer1->N, fp);
	vector_store(layer1->bf, layer1->N, fp);
	vector_store(layer1->bo, layer1->N, fp);

	vector_store(layer2->Wy, layer1->F * layer1->N, fp);
	vector_store(layer2->Wi, layer1->N * layer1->S, fp);
	vector_store(layer2->Wc, layer1->N * layer1->S, fp);
	vector_store(layer2->Wo, layer1->N * layer1->S, fp);
	vector_store(layer2->Wf, layer1->N * layer1->S, fp);

	vector_store(layer2->by, layer1->F, fp);
	vector_store(layer2->bi, layer1->N, fp);
	vector_store(layer2->bc, layer1->N, fp);
	vector_store(layer2->bf, layer1->N, fp);
	vector_store(layer2->bo, layer1->N, fp);

	fclose(fp);

}

void lstm_read_net(lstm_model_t* model, const char * filename) 
{
	FILE * fp;

	fp = fopen(filename, "r");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for reading.\n", filename);
		return;
	}

	vector_read(model->Wy, model->F * model->N, fp);
	vector_read(model->Wi, model->N * model->S, fp);
	vector_read(model->Wc, model->N * model->S, fp);
	vector_read(model->Wo, model->N * model->S, fp);
	vector_read(model->Wf, model->N * model->S, fp);

	vector_read(model->by, model->F, fp);
	vector_read(model->bi, model->N, fp);
	vector_read(model->bc, model->N, fp);
	vector_read(model->bf, model->N, fp);
	vector_read(model->bo, model->N, fp);

	printf("Loaded the net: %s\n", filename);
	fclose(fp);
}

void lstm_read_net_two_layers(lstm_model_t* layer1, lstm_model_t* layer2, const char * filename) 
{
	// Will only work for ( layer1->N, layer1->F ) == ( layer2->N, layer2->F )
	FILE * fp;

	fp = fopen(filename, "r");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for reading.\n", filename);
		return;
	}

	vector_read(layer1->Wy, layer1->F * layer1->N, fp);
	vector_read(layer1->Wi, layer1->N * layer1->S, fp);
	vector_read(layer1->Wc, layer1->N * layer1->S, fp);
	vector_read(layer1->Wo, layer1->N * layer1->S, fp);
	vector_read(layer1->Wf, layer1->N * layer1->S, fp);

	vector_read(layer1->by, layer1->F, fp);
	vector_read(layer1->bi, layer1->N, fp);
	vector_read(layer1->bc, layer1->N, fp);
	vector_read(layer1->bf, layer1->N, fp);
	vector_read(layer1->bo, layer1->N, fp);

	vector_read(layer2->Wy, layer1->F * layer1->N, fp);
	vector_read(layer2->Wi, layer1->N * layer1->S, fp);
	vector_read(layer2->Wc, layer1->N * layer1->S, fp);
	vector_read(layer2->Wo, layer1->N * layer1->S, fp);
	vector_read(layer2->Wf, layer1->N * layer1->S, fp);

	vector_read(layer2->by, layer1->F, fp);
	vector_read(layer2->bi, layer1->N, fp);
	vector_read(layer2->bc, layer1->N, fp);
	vector_read(layer2->bf, layer1->N, fp);
	vector_read(layer2->bo, layer1->N, fp);

	printf("Loaded the net: %s\n", filename);
	fclose(fp);
}


void lstm_output_string(lstm_model_t *model, set_T* char_index_mapping, char in, int length) 
{
	lstm_values_cache_t * cache;
	int i = 0, F = model->F, index, tmp_count = 0;
	char input = in;

	double first_layer_input[F];

	cache = lstm_cache_container_init(model->N, model->F);

	while ( i < length ) {
		index = set_char_to_indx(char_index_mapping,input);

		while ( tmp_count < F ){
			first_layer_input[tmp_count] = index == tmp_count ? 1.0 : 0.0;
			++tmp_count;
		}

		lstm_forward_propagate(model, first_layer_input, cache, cache, 1);
		input = set_probability_choice(char_index_mapping, cache->probs);
		printf ( "%c", input );
		++i;
	}

	lstm_cache_container_free(cache);
}

void lstm_output_string_two_layers(lstm_model_t *layer1, lstm_model_t *layer2, set_T* char_index_mapping, char in, int length) 
{
	lstm_values_cache_t * caches_layer_one, *caches_layer_two;
	int i = 0, count, index;
	char input = in;
	int F = layer1->F;

	caches_layer_one = lstm_cache_container_init(layer1->N, layer1->F);
	caches_layer_two = lstm_cache_container_init(layer2->N, layer2->F);

	double first_layer_input[F];

	while ( i < length ) {

		index = set_char_to_indx(char_index_mapping,input);

		count = 0;
		while ( count > F ) {
			first_layer_input[count] = count == index ? 1.0 : 0.0;
			++count;
		}

		lstm_forward_propagate(layer2, first_layer_input , caches_layer_two, caches_layer_two, 0);
		lstm_forward_propagate(layer1, caches_layer_two->probs , caches_layer_one, caches_layer_one, 1);
		input = set_probability_choice(char_index_mapping, caches_layer_one->probs);
		printf ( "%c", input );
		++i;
	}

	lstm_cache_container_free(caches_layer_one);
	lstm_cache_container_free(caches_layer_two);
}


void lstm_output_string_from_string(lstm_model_t *model, set_T* char_index_mapping, char * input_string, int out_length) 
{
	lstm_values_cache_t * cache;
	int i = 0, index, F, count;
	char input;
	size_t in_len = strlen(input_string);
	F = model->F;

	cache = lstm_cache_container_init(model->N, model->F);

	double first_layer_input[F];

	while ( i < in_len - 1 ) {

		index = set_char_to_indx(char_index_mapping,input_string[i]);

		count = 0;
		while ( count > F ) {
			first_layer_input[count] = count == index ? 1.0 : 0.0;
			++count;
		}

		lstm_forward_propagate(model, first_layer_input , cache, cache, 1);
		printf("%c", input_string[i]);
		++i;
	}

	input = input_string[i];

	printf("%c", input);
	i = 0;
	while ( i < out_length ) {

		index = set_char_to_indx(char_index_mapping,input);

		count = 0;
		while ( count > F ) {
			first_layer_input[count] = count == index ? 1.0 : 0.0;
			++count;
		}

		lstm_forward_propagate(model, first_layer_input, cache, cache,1);
		input = set_probability_choice(char_index_mapping, cache->probs);
		printf ( "%c", input );
		++i;
	}

	printf("\n");

	lstm_cache_container_free(cache);
}


void lstm_output_string_from_string_two_layers(lstm_model_t *layer1, lstm_model_t* layer2, set_T* char_index_mapping, char * input_string, int out_length) 
{
	lstm_values_cache_t *caches_layer_one, *caches_layer_two;
	int i = 0, count, index, in_len;
	char input;
	int F = layer1->F;

	caches_layer_one = lstm_cache_container_init(layer1->N, layer1->F);
	caches_layer_two = lstm_cache_container_init(layer2->N, layer2->F);

	double first_layer_input[F];

	in_len = strlen(input_string);

	while ( i < in_len - 1 ) {

		index = set_char_to_indx(char_index_mapping, input_string[i]);

		count = 0;
		while ( count > F ) {
			first_layer_input[count] = count == index ? 1.0 : 0.0;
			++count;
		}

		lstm_forward_propagate(layer2, first_layer_input , caches_layer_two, caches_layer_two, 0);
		lstm_forward_propagate(layer1, caches_layer_two->probs , caches_layer_one, caches_layer_one, 1);
		input = set_probability_choice(char_index_mapping, caches_layer_one->probs);
		printf ( "%c", input );
		++i;

	}

	input = input_string[i];

	printf("%c", input);
	i = 0;
	while ( i < out_length ) {
		index = set_char_to_indx(char_index_mapping,input);

		count = 0;
		while ( count > F ) {
			first_layer_input[count] = count == index ? 1.0 : 0.0;
			++count;
		}

		lstm_forward_propagate(layer2, first_layer_input , caches_layer_two, caches_layer_two, 0);
		lstm_forward_propagate(layer1, caches_layer_two->probs , caches_layer_one, caches_layer_one, 1);
		input = set_probability_choice(char_index_mapping, caches_layer_one->probs);
		printf ( "%c", input );
		++i;
	}

	printf("\n");

	lstm_cache_container_free(caches_layer_one);
	lstm_cache_container_free(caches_layer_two);
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

void lstm_model_regularization(lstm_model_t* model, lstm_model_t* gradients)
{
	double lambda = model->params->lambda; 

	vectors_add_scalar_multiply(gradients->Wy, model->Wy, model->F * model->N, lambda);
	vectors_add_scalar_multiply(gradients->Wi, model->Wi, model->N * model->S, lambda);
	vectors_add_scalar_multiply(gradients->Wc, model->Wc, model->N * model->S, lambda);
	vectors_add_scalar_multiply(gradients->Wo, model->Wo, model->N * model->S, lambda);
	vectors_add_scalar_multiply(gradients->Wf, model->Wf, model->N * model->S, lambda);

	vectors_add_scalar_multiply(gradients->by, model->by, model->F, lambda);
	vectors_add_scalar_multiply(gradients->bi, model->bi, model->N, lambda);
	vectors_add_scalar_multiply(gradients->bc, model->bc, model->N, lambda);
	vectors_add_scalar_multiply(gradients->bo, model->bo, model->N, lambda);
	vectors_add_scalar_multiply(gradients->bf, model->bf, model->N, lambda);
}

//						model, number of training points, X_train, Y_train, number of iterations
void lstm_train_the_net(lstm_model_t* model, set_T* char_index_mapping, unsigned int training_points, int* X_train, int* Y_train, unsigned long iterations)
{
	int N,F,S, status = 0;
	unsigned int i = 0, b = 0, q = 0, e1 = 0, e2 = 0, record_iteration = 0, tmp_count = 0;
	unsigned long n = 0, decrease_threshold = model->params->learning_rate_decrease_threshold;
	double loss = -1, loss_tmp = 0.0, record_keeper = 0.0;
	lstm_values_cache_t **caches, **tmp; 
	lstm_values_next_cache_t *d_next = NULL;
	lstm_model_t *gradients, *gradients_entry = NULL;

	N = model->N;
	F = model->F;
	S = model->S;

	double first_layer_input[F];

	caches = calloc(training_points + 1, sizeof(lstm_values_cache_t*));
	if ( caches == NULL ) 
		return;

	tmp = caches;

	lstm_init_model(F, N, &gradients, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);
	lstm_values_next_cache_init(&d_next, N, F);	
	lstm_init_model(F, N, &gradients_entry, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);

	i = 0;
	while ( i < training_points + 1){
		caches[i] = lstm_cache_container_init(N, F);
		++i;
	}

#ifdef GRADIENTS_CLIP
	unsigned long clip_count = 0;
#endif

	i = 0; b = 0;
	while ( n < iterations ){
		b = i;

		loss_tmp = 0.0;

		lstm_cache_container_set_start(caches[b]);

		q = 0;

		unsigned int check = i % training_points;

		while ( q < model->params->mini_batch_size ) {
			e1 = i % training_points;
			e2 = ( e1 + 1 ) % training_points;

			tmp_count = 0;
			while ( tmp_count < F ){
				first_layer_input[tmp_count] = tmp_count == X_train[e1] ? 1.0 : 0.0;
				++tmp_count;
			}

			lstm_forward_propagate(model, first_layer_input, caches[e1], caches[e2], 1);
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

		lstm_zero_d_next(d_next, F);

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

		assert(check == e2);

#ifdef MODEL_REGULARIZE
		lstm_model_regularization(model, gradients);
#endif

#ifdef GRADIENTS_CLIP
		if ( gradients_clip(gradients, model->params->gradient_clip_limit) ){
#ifdef DEBUG_PRINT	
				printf("Clipped the gradients at iteration: %lu\n", n);
#endif
			clip_count++;


#ifdef GRADIENT_CLIP_DECREASE_LR
			model->params->learning_rate *= 0.99;
			printf("New learning rate: %.20lf\n", model->params->learning_rate);
#endif				


			status = 1;
		} 
#endif
#ifdef GRADIENTS_FIT
		if ( gradients_fit(gradients, model->params->gradient_clip_limit) ){
			status = 1;
		} 	
#endif
		gradients_decend(model, gradients);

		if ( !( n % PRINT_EVERY_X_ITERATIONS ) ) {

			status = 0;
			printf("Iteration: %lu, Loss: %lf, record: %lf (iteration: %d)\n", n, loss, record_keeper, record_iteration);
#ifdef DEBUG_PRINT
			vector_print_min_max("Wy", model->Wy, F * N);
			vector_print_min_max("Wi", model->Wi, S * N);
			vector_print_min_max("Wc", model->Wc, S * N);
			vector_print_min_max("Wo", model->Wo, S * N);
			vector_print_min_max("Wf", model->Wf, S * N);
			vector_print_min_max("by", model->by, F);
			vector_print_min_max("bi", model->bi, N);
			vector_print_min_max("bc", model->bc, N);
			vector_print_min_max("bo", model->bo, N);
			vector_print_min_max("bf", model->bf, N);
#endif
			printf("=====================================================\n");

			lstm_output_string(model, char_index_mapping, X_train[b], NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING);

			printf("\n=====================================================\n");
			
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

#ifdef DECREASE_LR
		model->params->learning_rate = model->params->learning_rate / ( 1.0 + n / model->params->learning_rate_decrease );
#endif

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


//						model, number of training points, X_train, Y_train, number of iterations
void lstm_train_the_net_two_layers(lstm_model_t* model, lstm_model_t* layer1, lstm_model_t* layer2, set_T* char_index_mapping, unsigned int training_points, int* X_train, int* Y_train, unsigned long iterations)
{
	int N,F,S, status = 0;
	unsigned int i = 0, b = 0, q = 0, e1 = 0, e2 = 0, e3, record_iteration = 0, tmp_count;
	unsigned long n = 0, decrease_threshold = model->params->learning_rate_decrease_threshold;
	double loss = -1, loss_tmp = 0.0, record_keeper = 0.0;

	lstm_values_cache_t **caches_layer_one, **caches_layer_two; 

	lstm_values_next_cache_t *d_next_layer_one = NULL;
	lstm_values_next_cache_t *d_next_layer_two = NULL;
	
	lstm_model_t *gradients_layer_one, *gradients_entry_layer_one = NULL;
	lstm_model_t *gradients_layer_two, *gradients_entry_layer_two = NULL;

	N = model->N;
	F = model->F;
	S = model->S;

	double first_layer_input[F];

	caches_layer_one = calloc(training_points + 1, sizeof(lstm_values_cache_t*));
	if ( caches_layer_one == NULL ) 
		return;
	caches_layer_two = calloc(training_points + 1, sizeof(lstm_values_cache_t*));
	if ( caches_layer_two == NULL ) 
		return;

	lstm_init_model(F, N, &gradients_layer_one, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);
	lstm_values_next_cache_init(&d_next_layer_one, N, F);	
	lstm_init_model(F, N, &gradients_entry_layer_one, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);

	lstm_init_model(F, N, &gradients_layer_two, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);
	lstm_values_next_cache_init(&d_next_layer_two, N, F);	
	lstm_init_model(F, N, &gradients_entry_layer_two, YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);

	i = 0;
	while ( i < model->params->mini_batch_size ){
		caches_layer_one[i] = lstm_cache_container_init(N, F);
		caches_layer_two[i] = lstm_cache_container_init(N, F);
		++i;
	}

#ifdef GRADIENTS_CLIP
	unsigned long clip_count = 0;
#endif

	i = 0; b = 0;
	while ( n < iterations ){
		b = i;

		loss_tmp = 0.0;

		lstm_cache_container_set_start(caches_layer_one[0]);
		lstm_cache_container_set_start(caches_layer_two[0]);

		q = 0;

		unsigned int check = i % training_points;

		while ( q < model->params->mini_batch_size ) {
			e1 = q % model->params->mini_batch_size;
			e2 = ( e1 + 1 ) % model->params->mini_batch_size;
			
			e3 = i % training_points;

			tmp_count = 0;
			while ( tmp_count < F ){
				first_layer_input[tmp_count] = tmp_count == X_train[e3] ? 1.0 : 0.0;
				++tmp_count;
			}
			/* Layer numbering starts at the output point of the net */
			lstm_forward_propagate(layer2, first_layer_input, caches_layer_two[e1], caches_layer_two[e2], 0);
			lstm_forward_propagate(layer1, caches_layer_two[e2]->probs, caches_layer_one[e1], caches_layer_one[e2], 1);
			loss_tmp += cross_entropy( caches_layer_one[e2]->probs, Y_train[e3]);
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
			record_iteration = n;
		}

		lstm_zero_the_model(gradients_layer_one);
		lstm_zero_the_model(gradients_layer_two);

		lstm_zero_d_next(d_next_layer_one, F);
		lstm_zero_d_next(d_next_layer_two, F);
 
		while ( q > 0 ) {
			e1 = q % model->params->mini_batch_size;
			e2 = ( model->params->mini_batch_size + e1 - 1 ) % model->params->mini_batch_size;

			e3 = ( training_points + i - 1 ) % training_points;

			lstm_zero_the_model(gradients_entry_layer_one);
			lstm_zero_the_model(gradients_entry_layer_two);

			lstm_backward_propagate(layer1, caches_layer_one[e1]->probs, Y_train[e3], d_next_layer_one, caches_layer_one[e1], gradients_entry_layer_one, d_next_layer_one);
			lstm_backward_propagate(layer2, d_next_layer_one->dldY_pass, -1 , d_next_layer_two, caches_layer_two[e1], gradients_entry_layer_two, d_next_layer_two);

			//gradients_fit(gradients_entry, model->params->gradient_clip_limit);

			sum_gradients(gradients_layer_one, gradients_entry_layer_one);
			sum_gradients(gradients_layer_two, gradients_entry_layer_two);

			i--; q--;
		}

		assert(check == e3);


#ifdef GRADIENTS_CLIP
		if ( gradients_clip(gradients_layer_one, model->params->gradient_clip_limit) ){
#ifdef DEBUG_PRINT	
				printf("Clipped the gradients at iteration: %lu\n", n);
#endif
			clip_count++;


#ifdef GRADIENT_CLIP_DECREASE_LR
			model->params->learning_rate *= 0.99;
			printf("New learning rate: %.20lf\n", model->params->learning_rate);
#endif				


			status = 1;
		} 
#endif

#ifdef GRADIENTS_CLIP
		if ( gradients_clip(gradients_layer_two, model->params->gradient_clip_limit) ){
#ifdef DEBUG_PRINT	
				printf("Clipped the gradients at iteration: %lu\n", n);
#endif
			clip_count++;
			
			status = 1;
		} 
#endif


#ifdef GRADIENTS_FIT
		if ( gradients_fit(gradients_layer_one, model->params->gradient_clip_limit) ){
			status = 1;
		} 	
#endif

#ifdef GRADIENTS_FIT
		if ( gradients_fit(gradients_layer_two, model->params->gradient_clip_limit) ){
			status = 1;
		} 	
#endif

		gradients_decend(layer1, gradients_layer_one);
		gradients_decend(layer2, gradients_layer_two);

		if ( !( n % PRINT_EVERY_X_ITERATIONS ) ) {

			status = 0;
			printf("Iteration: %lu, Loss: %lf, record: %lf (iteration: %d)\n", n, loss, record_keeper, record_iteration);
			printf("=====================================================\n");

			lstm_output_string_two_layers(layer1, layer2, char_index_mapping, X_train[b], NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING);

			printf("\n=====================================================\n");
			
			// Flushing stdout
			fflush(stdout);
		}

		if ( !(n % STORE_EVERY_X_ITERATIONS ) && n > 0 )
			lstm_store_net_two_layers(layer1, layer2, STD_LOADABLE_NET_NAME);

		if ( !(n % STORE_PROGRESS_EVERY_X_ITERATIONS ))
			lstm_store_progress(n, loss);


		i = b + model->params->mini_batch_size;
		if ( i >= training_points )
			i = 0;

#ifdef DECREASE_LR
		model->params->learning_rate = model->params->learning_rate / ( 1.0 + n / model->params->learning_rate_decrease );
#endif

		++n;
	}

	lstm_values_next_cache_free(d_next_layer_one);
	lstm_values_next_cache_free(d_next_layer_two);

	i = 0;
	while ( i < model->params->mini_batch_size) {
		lstm_cache_container_free(caches_layer_one[i]);
		free(caches_layer_one[i]);
		lstm_cache_container_free(caches_layer_two[i]);
		free(caches_layer_two[i]);
		++i;
	}

	lstm_free_model(gradients_entry_layer_one);
	lstm_free_model(gradients_layer_one);
	lstm_free_model(gradients_entry_layer_two);
	lstm_free_model(gradients_layer_two);

	free(caches_layer_one);
	free(caches_layer_two);

}







