#include "lstm.h"

void lstm_init_fail(const char * msg)
{
	printf("%s",msg);
	exit(-1);
}

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

void lstm_values_state_init(lstm_values_state_t** d_next_to_set, int N)
{
	lstm_values_state_t * d_next = calloc(1, sizeof(lstm_values_state_t));
	if ( d_next == NULL )
		return;
	init_zero_vector(&d_next->c, N);
	init_zero_vector(&d_next->h, N);

	*d_next_to_set = d_next;
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


void gradients_adam_optimizer(lstm_model_t* model, lstm_model_t* gradients, lstm_model_t* M, lstm_model_t* R, unsigned int t) 
{
	double beta1 = model->params->beta1;
	double beta2 = model->params->beta2;

	double beta1t = 1.0 / ( 1.0 - pow(beta1, t+1));
	double beta2t = 1.0 / ( 1.0 - pow(beta2, t+1));

	if ( !(beta2t == beta2t) ) {
		printf("beta2t: %lf\n", beta2t);
		exit(0);
	}

	copy_vector(gradients->Wym, gradients->Wy, model->F * model->N);
	copy_vector(gradients->Wim, gradients->Wi, model->N * model->S);
	copy_vector(gradients->Wcm, gradients->Wc, model->N * model->S);
	copy_vector(gradients->Wom, gradients->Wo, model->N * model->S);
	copy_vector(gradients->Wfm, gradients->Wf, model->N * model->S);

	copy_vector(gradients->bym, gradients->by, model->F);
	copy_vector(gradients->bim, gradients->bi, model->N);
	copy_vector(gradients->bcm, gradients->bc, model->N);
	copy_vector(gradients->bom, gradients->bo, model->N);
	copy_vector(gradients->bfm, gradients->bf, model->N);

	vectors_mutliply_scalar(gradients->Wym, 1.0 - beta1, model->F * model->N);
	vectors_mutliply_scalar(gradients->Wim, 1.0 - beta1, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wcm, 1.0 - beta1, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wom, 1.0 - beta1, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wfm, 1.0 - beta1, model->N * model->S);

	vectors_mutliply_scalar(gradients->bym, 1.0 - beta1, model->F);
	vectors_mutliply_scalar(gradients->bim, 1.0 - beta1, model->N);
	vectors_mutliply_scalar(gradients->bcm, 1.0 - beta1, model->N);
	vectors_mutliply_scalar(gradients->bom, 1.0 - beta1, model->N);
	vectors_mutliply_scalar(gradients->bfm, 1.0 - beta1, model->N);

	vectors_mutliply_scalar(M->Wy, beta1, model->F * model->N);
	vectors_mutliply_scalar(M->Wi, beta1, model->N * model->S);
	vectors_mutliply_scalar(M->Wc, beta1, model->N * model->S);
	vectors_mutliply_scalar(M->Wo, beta1, model->N * model->S);
	vectors_mutliply_scalar(M->Wf, beta1, model->N * model->S);

	vectors_mutliply_scalar(M->by, beta1, model->F);
	vectors_mutliply_scalar(M->bi, beta1, model->N);
	vectors_mutliply_scalar(M->bc, beta1, model->N);
	vectors_mutliply_scalar(M->bo, beta1, model->N);
	vectors_mutliply_scalar(M->bf, beta1, model->N);

	vectors_add(M->Wy, gradients->Wy, model->F * model->N);
	vectors_add(M->Wi, gradients->Wi, model->N * model->S);
	vectors_add(M->Wc, gradients->Wc, model->N * model->S);
	vectors_add(M->Wo, gradients->Wo, model->N * model->S);
	vectors_add(M->Wf, gradients->Wf, model->N * model->S);

	vectors_add(M->by, gradients->by, model->F);
	vectors_add(M->bi, gradients->bi, model->N);
	vectors_add(M->bc, gradients->bc, model->N);
	vectors_add(M->bo, gradients->bo, model->N);
	vectors_add(M->bf, gradients->bf, model->N);
	
	// M Done!
	// Computing R

	vectors_multiply(gradients->Wy, gradients->Wy, model->F * model->N);
	vectors_multiply(gradients->Wi, gradients->Wi, model->N * model->S);
	vectors_multiply(gradients->Wc, gradients->Wc, model->N * model->S);
	vectors_multiply(gradients->Wo, gradients->Wo, model->N * model->S);
	vectors_multiply(gradients->Wf, gradients->Wf, model->N * model->S);

	vectors_multiply(gradients->by, gradients->by, model->F );
	vectors_multiply(gradients->bi, gradients->bi, model->N );
	vectors_multiply(gradients->bc, gradients->bc, model->N );
	vectors_multiply(gradients->bo, gradients->bo, model->N );
	vectors_multiply(gradients->bf, gradients->bf, model->N );

	copy_vector(gradients->Wym, gradients->Wy, model->F * model->N);
	copy_vector(gradients->Wim, gradients->Wi, model->N * model->S);
	copy_vector(gradients->Wcm, gradients->Wc, model->N * model->S);
	copy_vector(gradients->Wom, gradients->Wo, model->N * model->S);
	copy_vector(gradients->Wfm, gradients->Wf, model->N * model->S);

	copy_vector(gradients->bym, gradients->by, model->F);
	copy_vector(gradients->bim, gradients->bi, model->N);
	copy_vector(gradients->bcm, gradients->bc, model->N);
	copy_vector(gradients->bom, gradients->bo, model->N);
	copy_vector(gradients->bfm, gradients->bf, model->N);

	vectors_mutliply_scalar(gradients->Wym, 1.0 - beta2, model->F * model->N);
	vectors_mutliply_scalar(gradients->Wim, 1.0 - beta2, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wcm, 1.0 - beta2, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wom, 1.0 - beta2, model->N * model->S);
	vectors_mutliply_scalar(gradients->Wfm, 1.0 - beta2, model->N * model->S);

	vectors_mutliply_scalar(gradients->bym, 1.0 - beta2, model->F);
	vectors_mutliply_scalar(gradients->bim, 1.0 - beta2, model->N);
	vectors_mutliply_scalar(gradients->bcm, 1.0 - beta2, model->N);
	vectors_mutliply_scalar(gradients->bom, 1.0 - beta2, model->N);
	vectors_mutliply_scalar(gradients->bfm, 1.0 - beta2, model->N);

	vectors_mutliply_scalar(R->Wy, beta2, model->F * model->N);
	vectors_mutliply_scalar(R->Wi, beta2, model->N * model->S);
	vectors_mutliply_scalar(R->Wc, beta2, model->N * model->S);
	vectors_mutliply_scalar(R->Wo, beta2, model->N * model->S);
	vectors_mutliply_scalar(R->Wf, beta2, model->N * model->S);

	vectors_mutliply_scalar(R->by, beta2, model->F);
	vectors_mutliply_scalar(R->bi, beta2, model->N);
	vectors_mutliply_scalar(R->bc, beta2, model->N);
	vectors_mutliply_scalar(R->bo, beta2, model->N);
	vectors_mutliply_scalar(R->bf, beta2, model->N);

	vectors_add(R->Wy, gradients->Wy, model->F * model->N);
	vectors_add(R->Wi, gradients->Wi, model->N * model->S);
	vectors_add(R->Wc, gradients->Wc, model->N * model->S);
	vectors_add(R->Wo, gradients->Wo, model->N * model->S);
	vectors_add(R->Wf, gradients->Wf, model->N * model->S);

	vectors_add(R->by, gradients->by, model->F);
	vectors_add(R->bi, gradients->bi, model->N);
	vectors_add(R->bc, gradients->bc, model->N);
	vectors_add(R->bo, gradients->bo, model->N);
	vectors_add(R->bf, gradients->bf, model->N);

	// R done!

	copy_vector(M->Wym, M->Wy, model->F * model->N);
	copy_vector(M->Wim, M->Wi, model->N * model->S);
	copy_vector(M->Wcm, M->Wc, model->N * model->S);
	copy_vector(M->Wom, M->Wo, model->N * model->S);
	copy_vector(M->Wfm, M->Wf, model->N * model->S);

	copy_vector(M->bym, M->by, model->F);
	copy_vector(M->bim, M->bi, model->N);
	copy_vector(M->bcm, M->bc, model->N);
	copy_vector(M->bom, M->bo, model->N);
	copy_vector(M->bfm, M->bf, model->N);

	vectors_mutliply_scalar(M->Wym, beta1t, model->F * model->N);
	vectors_mutliply_scalar(M->Wim, beta1t, model->N * model->S);
	vectors_mutliply_scalar(M->Wcm, beta1t, model->N * model->S);
	vectors_mutliply_scalar(M->Wom, beta1t, model->N * model->S);
	vectors_mutliply_scalar(M->Wfm, beta1t, model->N * model->S);

	vectors_mutliply_scalar(M->bym, beta1t, model->F);
	vectors_mutliply_scalar(M->bim, beta1t, model->N);
	vectors_mutliply_scalar(M->bcm, beta1t, model->N);
	vectors_mutliply_scalar(M->bom, beta1t, model->N);
	vectors_mutliply_scalar(M->bfm, beta1t, model->N);

	// M hat done!

	copy_vector(R->Wym, R->Wy, model->F * model->N);
	copy_vector(R->Wim, R->Wi, model->N * model->S);
	copy_vector(R->Wcm, R->Wc, model->N * model->S);
	copy_vector(R->Wom, R->Wo, model->N * model->S);
	copy_vector(R->Wfm, R->Wf, model->N * model->S);

	copy_vector(R->bym, R->by, model->F);
	copy_vector(R->bim, R->bi, model->N);
	copy_vector(R->bcm, R->bc, model->N);
	copy_vector(R->bom, R->bo, model->N);
	copy_vector(R->bfm, R->bf, model->N);

	vectors_mutliply_scalar(R->Wym, beta2t, model->F * model->N);
	vectors_mutliply_scalar(R->Wim, beta2t, model->N * model->S);
	vectors_mutliply_scalar(R->Wcm, beta2t, model->N * model->S);
	vectors_mutliply_scalar(R->Wom, beta2t, model->N * model->S);
	vectors_mutliply_scalar(R->Wfm, beta2t, model->N * model->S);

	vectors_mutliply_scalar(R->bym, beta2t, model->F);
	vectors_mutliply_scalar(R->bim, beta2t, model->N);
	vectors_mutliply_scalar(R->bcm, beta2t, model->N);
	vectors_mutliply_scalar(R->bom, beta2t, model->N);
	vectors_mutliply_scalar(R->bfm, beta2t, model->N);

	// R hat done!

	vector_sqrt(R->Wym, model->F * model->N);
	vector_sqrt(R->Wim, model->N * model->S);
	vector_sqrt(R->Wcm, model->N * model->S);
	vector_sqrt(R->Wom, model->N * model->S);
	vector_sqrt(R->Wfm, model->N * model->S);

	vector_sqrt(R->bym, model->F);
	vector_sqrt(R->bim, model->N);
	vector_sqrt(R->bcm, model->N);
	vector_sqrt(R->bom, model->N);
	vector_sqrt(R->bfm, model->N);

	vectors_add_scalar(R->Wym, 1e-7, model->F * model->N);
	vectors_add_scalar(R->Wim, 1e-7, model->N * model->S);
	vectors_add_scalar(R->Wcm, 1e-7, model->N * model->S);
	vectors_add_scalar(R->Wom, 1e-7, model->N * model->S);
	vectors_add_scalar(R->Wfm, 1e-7, model->N * model->S);

	vectors_add_scalar(R->bym, 1e-7, model->F);
	vectors_add_scalar(R->bim, 1e-7, model->N);
	vectors_add_scalar(R->bcm, 1e-7, model->N);
	vectors_add_scalar(R->bom, 1e-7, model->N);
	vectors_add_scalar(R->bfm, 1e-7, model->N);

	copy_vector(gradients->Wym, M->Wym, model->F * model->N);
	copy_vector(gradients->Wim, M->Wim, model->N * model->S);
	copy_vector(gradients->Wcm, M->Wcm, model->N * model->S);
	copy_vector(gradients->Wom, M->Wom, model->N * model->S);
	copy_vector(gradients->Wfm, M->Wfm, model->N * model->S);

	copy_vector(gradients->bym, M->bym, model->F);
	copy_vector(gradients->bim, M->bim, model->N);
	copy_vector(gradients->bcm, M->bcm, model->N);
	copy_vector(gradients->bom, M->bom, model->N);
	copy_vector(gradients->bfm, M->bfm, model->N);	

	vectors_scalar_multiply(gradients->Wym, model->params->learning_rate, model->F * model->N);
	vectors_scalar_multiply(gradients->Wim, model->params->learning_rate, model->N * model->S);
	vectors_scalar_multiply(gradients->Wcm, model->params->learning_rate, model->N * model->S);
	vectors_scalar_multiply(gradients->Wom, model->params->learning_rate, model->N * model->S);
	vectors_scalar_multiply(gradients->Wfm, model->params->learning_rate, model->N * model->S);

	vectors_scalar_multiply(gradients->bym, model->params->learning_rate, model->F);
	vectors_scalar_multiply(gradients->bim, model->params->learning_rate, model->N);
	vectors_scalar_multiply(gradients->bcm, model->params->learning_rate, model->N);
	vectors_scalar_multiply(gradients->bom, model->params->learning_rate, model->N);
	vectors_scalar_multiply(gradients->bfm, model->params->learning_rate, model->N);	

	vectors_div(gradients->Wym, R->Wym, model->F * model->N);
	vectors_div(gradients->Wim, R->Wim, model->N * model->S);
	vectors_div(gradients->Wcm, R->Wcm, model->N * model->S);
	vectors_div(gradients->Wom, R->Wom, model->N * model->S);
	vectors_div(gradients->Wfm, R->Wfm, model->N * model->S);

	vectors_div(gradients->bym, R->bym, model->F);
	vectors_div(gradients->bim, R->bim, model->N);
	vectors_div(gradients->bcm, R->bcm, model->N);
	vectors_div(gradients->bom, R->bom, model->N);
	vectors_div(gradients->bfm, R->bfm, model->N);


	vectors_substract(model->Wy, gradients->Wym, model->F * model->N);
	vectors_substract(model->Wi, gradients->Wim, model->N * model->S);
	vectors_substract(model->Wc, gradients->Wcm, model->N * model->S);
	vectors_substract(model->Wo, gradients->Wom, model->N * model->S);
	vectors_substract(model->Wf, gradients->Wfm, model->N * model->S);

	vectors_substract(model->by, gradients->bym, model->F);
	vectors_substract(model->bi, gradients->bim, model->N);
	vectors_substract(model->bc, gradients->bcm, model->N);
	vectors_substract(model->bo, gradients->bom, model->N);
	vectors_substract(model->bf, gradients->bfm, model->N);	

/*			try:
				M[k]
				R[k]
			except KeyError:
				M[k] = 0.
				R[k] = 0.

			M[k] = exp_running_avg(M[k], grad[k], beta1)
			R[k] = exp_running_avg(R[k], grad[k]**2, beta2)

			m_k_hat = M[k] / (1. - beta1**(t))
			r_k_hat = R[k] / (1. - beta2**(t))

			model[k] -= alpha * m_k_hat / (np.sqrt(r_k_hat) + 1e-7)

*/
	
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

void lstm_values_next_state_free(lstm_values_state_t* d_next)
{
	free_vector(&d_next->h);
	free_vector(&d_next->c);
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
		softmax_layers_forward(cache_out->probs, cache_out->probs, F, model->params->softmax_temp);
	} 
#ifdef INTERLAYER_SIGMOID_ACTIVATION
 	else {
		sigmoid_forward(cache_out->probs, cache_out->probs, F);
		copy_vector(cache_out->probs_before_sigma, cache_out->probs, F);
	} 
#endif

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
	}
#ifdef INTERLAYER_SIGMOID_ACTIVATION
	else {
		sigmoid_backward(dldy, cache_in->probs_before_sigma, dldy, F);
	}
#endif
	
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

	vector_set_to_zero(model->Wym, model->F * model->N);
	vector_set_to_zero(model->Wim, model->N * model->S);
	vector_set_to_zero(model->Wcm, model->N * model->S);
	vector_set_to_zero(model->Wom, model->N * model->S);
	vector_set_to_zero(model->Wfm, model->N * model->S);

	vector_set_to_zero(model->bym, model->F);
	vector_set_to_zero(model->bim, model->N);
	vector_set_to_zero(model->bcm, model->N);
	vector_set_to_zero(model->bfm, model->N);
	vector_set_to_zero(model->bom, model->N);

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

void lstm_next_state_copy(lstm_values_state_t * state, lstm_values_cache_t * cache, int write )
{
	if ( write ) {
		// Write to the state carrying unit
		copy_vector(state->h, cache->h, NEURONS);
		copy_vector(state->c, cache->c, NEURONS);
	} else {
		// Withdraw from the state carrying unit
		copy_vector(cache->h, state->h, NEURONS);
		copy_vector(cache->c, state->c, NEURONS);
	}

}

void lstm_cache_container_set_start(lstm_values_cache_t * cache)
{
	// State variables set to zero
	vector_set_to_zero(cache->h, NEURONS); 
	vector_set_to_zero(cache->c, NEURONS); 

}

void lstm_store_net_layers(lstm_model_t** model, const char * filename) 
{
	FILE * fp;
	int p = 0;

	fp = fopen(filename, "w");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for writing.\n", filename);
		return;
	}

	while ( p < LAYERS ) {

		vector_store(model[p]->Wy, model[p]->F * model[p]->N, fp);
		vector_store(model[p]->Wi, model[p]->N * model[p]->S, fp);
		vector_store(model[p]->Wc, model[p]->N * model[p]->S, fp);
		vector_store(model[p]->Wo, model[p]->N * model[p]->S, fp);
		vector_store(model[p]->Wf, model[p]->N * model[p]->S, fp);

		vector_store(model[p]->by, model[p]->F, fp);
		vector_store(model[p]->bi, model[p]->N, fp);
		vector_store(model[p]->bc, model[p]->N, fp);
		vector_store(model[p]->bf, model[p]->N, fp);
		vector_store(model[p]->bo, model[p]->N, fp);

		++p;
	}

	fclose(fp);

}

void lstm_store_net_layers_as_json(lstm_model_t** model, const char * filename, set_T *set) 
{
	FILE * fp;
	int p = 0, r = 0, c = 0;

	fp = fopen(filename, "w");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for writing.\n", filename);
		return;
	}

	fprintf(fp, "{\n\"Feature mapping\": ");
	set_store_as_json(set, fp);

	fprintf(fp, ",\n\"LSTM layers\": %d,\n", LAYERS);

	while ( p < LAYERS ) {

		if ( p > 0 ) 
			fprintf(fp, ",\n");

		fprintf(fp, "\"Layer %d\": {\n", p+1);

		fprintf(fp, "\t\"Wy\": ");
		vector_store_as_matrix_json(model[p]->Wy, model[p]->N, model[p]->F, fp);
		fprintf(fp, ",\n\t\"Wi\": ");
		vector_store_as_matrix_json(model[p]->Wi, model[p]->N, model[p]->S, fp);
		fprintf(fp, ",\n\t\"Wc\": ");
		vector_store_as_matrix_json(model[p]->Wc, model[p]->N, model[p]->S, fp);
		fprintf(fp, ",\n\t\"Wo\": ");
		vector_store_as_matrix_json(model[p]->Wo, model[p]->N, model[p]->S, fp);
		fprintf(fp, ",\n\t\"Wf\": ");
		vector_store_as_matrix_json(model[p]->Wf, model[p]->N, model[p]->S, fp);

		fprintf(fp, ",\n\t\"by\": ");
		vector_store_json(model[p]->by, model[p]->F, fp);
		fprintf(fp, ",\n\t\"bi\": ");
		vector_store_json(model[p]->bi, model[p]->N, fp);
		fprintf(fp, ",\n\t\"bc\": ");
		vector_store_json(model[p]->bc, model[p]->N, fp);
		fprintf(fp, ",\n\t\"bf\": ");
		vector_store_json(model[p]->bf, model[p]->N, fp);
		fprintf(fp, ",\n\t\"bo\": ");
		vector_store_json(model[p]->bo, model[p]->N, fp);

		fprintf(fp, "}\n");

		++p;
	}

	fprintf(fp, "}\n");

	fclose(fp);

}


void lstm_read_net_layers(lstm_model_t** model, const char * filename) 
{
	// Will only work for ( layer1->N, layer1->F ) == ( layer2->N, layer2->F )
	FILE * fp;

	int p = 0;

	fp = fopen(filename, "r");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for reading.\n", filename);
		return;
	}

	while ( p < LAYERS ) {

		vector_read(model[p]->Wy, model[p]->F * model[p]->N, fp);
		vector_read(model[p]->Wi, model[p]->N * model[p]->S, fp);
		vector_read(model[p]->Wc, model[p]->N * model[p]->S, fp);
		vector_read(model[p]->Wo, model[p]->N * model[p]->S, fp);
		vector_read(model[p]->Wf, model[p]->N * model[p]->S, fp);

		vector_read(model[p]->by, model[p]->F, fp);
		vector_read(model[p]->bi, model[p]->N, fp);
		vector_read(model[p]->bc, model[p]->N, fp);
		vector_read(model[p]->bf, model[p]->N, fp);
		vector_read(model[p]->bo, model[p]->N, fp);

		++p;	
	}

	printf("Loaded the net: %s\n", filename);
	fclose(fp);
}

void lstm_output_string_layers(lstm_model_t ** model_layers, set_T* char_index_mapping, int first, int numbers_to_display, int layers)
{
	lstm_values_cache_t ***caches_layer;
	int i = 0, count, index, p = 0, b = 0;
	char input = set_char_to_indx(char_index_mapping, first);
	int F = model_layers[0]->F;

	caches_layer = calloc(layers, sizeof(lstm_values_cache_t**));

	if ( caches_layer == NULL )
		lstm_init_fail("Failed to output string\n");

	p = 0;
	while ( p < layers ) {
		caches_layer[p] = calloc(2, sizeof(lstm_values_cache_t*));
		b = 0;
		while ( b < 2 ) {
			caches_layer[p][b] = lstm_cache_container_init(model_layers[p]->N, model_layers[p]->F); 
			++b;
		}
		++p;
	}

	double first_layer_input[F];

	lstm_cache_container_set_start(caches_layer[0][0]);
	lstm_cache_container_set_start(caches_layer[0][0]);

	while ( i < numbers_to_display ) {

		index = set_char_to_indx(char_index_mapping,input);

		count = 0;
		while ( count < F ) {
			first_layer_input[count] = 0.0;
			++count;
		}

		first_layer_input[index] = 1.0;

		p = layers - 1;
		lstm_forward_propagate(model_layers[p], first_layer_input, caches_layer[p][i % 2], caches_layer[p][(i+1)%2], p == 0);

		if ( p > 0 ) {
			--p;
			while ( p >= 0 ) {
				lstm_forward_propagate(model_layers[p], caches_layer[p+1][(i+1)%2]->probs, caches_layer[p][i % 2], caches_layer[p][(i+1)%2], p == 0);	
				--p;
			}
			p = 0;
		}

		input = set_probability_choice(char_index_mapping, caches_layer[p][(i+1)%2]->probs);
		printf ( "%c", input );

		++i;
	}

	p = 0;
	while ( p < layers ) {

		b = 0;
		while ( b < 2 ) {
			lstm_cache_container_free( caches_layer[p][b]);
			free(caches_layer[p][b]);
			++b;
		}
		free(caches_layer[p]);
		++p;
	}

	free(caches_layer);
}

void lstm_output_string_from_string_layers(lstm_model_t **model_layers, set_T* char_index_mapping, char * input_string, int out_length) 
{
	lstm_values_cache_t ***caches_layers;
	int i = 0, count, index, in_len;
	char input;
	int F = model_layers[0]->F;

	int layers = LAYERS;

	int p = 0;

	caches_layers = calloc(layers, sizeof(lstm_values_cache_t**));

	if ( caches_layers == NULL ) {
		printf("%s Error: calloc failed \n", __func__);
		exit(-1);
	}

	while ( p < LAYERS ) {
		caches_layers[p] = calloc(2, sizeof(lstm_values_cache_t*));
		
		i = 0; 
		while ( i < 2 ){
			caches_layers[p][i] = lstm_cache_container_init(model_layers[0]->N, model_layers[0]->F);
			++i;
		}

		++p;
	}

	double first_layer_input[F];

	in_len = strlen(input_string);
	i = 0;

	while ( i < in_len ) {
		printf("%c", input_string[i]);
		index = set_char_to_indx(char_index_mapping, input_string[i]);

		count = 0;
		while ( count < F ) {
			first_layer_input[count] = count == index ? 1.0 : 0.0;
			++count;
		}

		p = layers - 1;
		lstm_forward_propagate(model_layers[p], first_layer_input, caches_layers[p][i%2], caches_layers[p][(i+1)%2], p == 0);

		if ( p > 0 ) {
			--p;
			while ( p >= 0 ) {
				lstm_forward_propagate(model_layers[p], caches_layers[p+1][(i+1)%2]->probs, caches_layers[p][i%2], caches_layers[p][(i+1)%2], p == 0);	
				--p;
			}
			p = 0;
		}

		++i;

	}

	input = set_probability_choice(char_index_mapping, caches_layers[0][i%2]->probs);

	printf("%c", input);
	i = 0;
	while ( i < out_length ) {
		index = set_char_to_indx(char_index_mapping,input);

		count = 0;
		while ( count < F ) {
			first_layer_input[count] = count == index ? 1.0 : 0.0;
			++count;
		}

		p = layers - 1;
		lstm_forward_propagate(model_layers[p], first_layer_input, caches_layers[p][i%2], caches_layers[p][(i+1)%2], p == 0);

		if ( p > 0 ) {
			--p;
			while ( p >= 0 ) {
				lstm_forward_propagate(model_layers[p], caches_layers[p+1][ (i+1) % 2 ]->probs, caches_layers[p][i%2], caches_layers[p][(i+1)%2], p == 0);	
				--p;
			}
			p = 0;
		}
		input = set_probability_choice(char_index_mapping, caches_layers[p][(i+1)%2]->probs);
		printf ( "%c", input );
//		set_print(char_index_mapping,caches_layer_one->probs);
		++i;
	}

	printf("\n");

	p = 0;
	while ( p < LAYERS ) {

		i = 0; 
		while ( i < 2 ){
			lstm_cache_container_free( caches_layers[p][i] ); 
			free(caches_layers[p][i]);
			++i;
		}

		free(caches_layers[p]);

		++p;
	}

	free(caches_layers);
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
void lstm_train(lstm_model_t* model, lstm_model_t** model_layers, set_T* char_index_mapping, unsigned int training_points, int* X_train, int* Y_train, unsigned long iterations, int layers)
{
	int N,F,S, status = 0, p = 0;
	unsigned int i = 0, b = 0, q = 0, e1 = 0, e2 = 0, e3, record_iteration = 0, tmp_count, trailing;
	unsigned long n = 0, decrease_threshold = model->params->learning_rate_decrease_threshold, epoch = 0;
	double loss = -1, loss_tmp = 0.0, record_keeper = 0.0;
	double initial_learning_rate = model->params->learning_rate;
	time_t time_iter;
	char time_buffer[40];

	lstm_values_cache_t ***cache_layers;

	lstm_values_next_cache_t **d_next_layers;
	
	lstm_model_t **gradient_layers, **gradient_layers_entry,  **M_layers, **R_layers;

	N = model->N;
	F = model->F;
	S = model->S;

	double first_layer_input[F];

#ifdef STATEFUL
	lstm_values_state_t ** stateful_d_next;
	stateful_d_next = calloc(layers, sizeof(lstm_values_state_t*));
	if ( stateful_d_next == NULL )
		lstm_init_fail("Failed to allocate memory for stateful backprop through time deltas\n");
	i = 0;
	while ( i < layers) {
		stateful_d_next[i] = calloc( training_points/model->params->mini_batch_size + 1, sizeof(lstm_values_state_t));
		if ( stateful_d_next[i] == NULL )
			lstm_init_fail("Failed to allocate memory for stateful backprop through time deltas, inner in layer\n");
		lstm_values_state_init(&stateful_d_next[i], N);
		++i;
	}
#endif

	i = 0;
	cache_layers = calloc(layers, sizeof(lstm_values_cache_t**));

	if ( cache_layers == NULL )
		lstm_init_fail("Failed to allocate memory for the caches\n");

	while ( i < layers ) {
		cache_layers[i] = calloc(model->params->mini_batch_size + 1, sizeof(lstm_values_cache_t*));
		if ( cache_layers[i] == NULL )
			lstm_init_fail("Failed to allocate memory for the caches\n");

		p = 0;
		while ( p < model->params->mini_batch_size + 1 ){
			cache_layers[i][p] = lstm_cache_container_init(N, F);
			if ( cache_layers[i][p] == NULL )
				lstm_init_fail("Failed to allocate memory for the caches\n");		
			++p;
		}

		++i;
	}

	gradient_layers = calloc(layers, sizeof(lstm_model_t*) );
	if ( gradient_layers == NULL )
		lstm_init_fail("Failed to allocate memory for gradients\n");

	gradient_layers_entry = calloc(layers, sizeof(lstm_model_t*) );
	if ( gradient_layers_entry == NULL )
		lstm_init_fail("Failed to allocate memory for gradients\n");

	d_next_layers = calloc(layers, sizeof(lstm_values_next_cache_t *));
	if ( d_next_layers == NULL )
		lstm_init_fail("Failed to allocate memory for backprop through time deltas\n");

	if ( model->params->optimizer == OPTIMIZE_ADAM ) {

		M_layers = calloc(layers, sizeof(lstm_model_t*) );
		R_layers = calloc(layers, sizeof(lstm_model_t*) );

	}

	if ( M_layers == NULL || R_layers == NULL ) {
		lstm_init_fail("Failed to init M or R dicts for adam optimization\n");
	}

	i = 0;
	while ( i < layers ) {
		lstm_init_model(F, N, &gradient_layers[i], YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);
		lstm_init_model(F, N, &gradient_layers_entry[i], YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);
		lstm_values_next_cache_init(&d_next_layers[i], N, F);

		if ( model->params->optimizer == OPTIMIZE_ADAM ) {
			lstm_init_model(F, N, &M_layers[i], YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);
			lstm_init_model(F, N, &R_layers[i], YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE, model->params);
		}

		++i;
	}

	i = 0; b = 0;
	while ( n < iterations ){
		b = i;

		loss_tmp = 0.0;

		q = 0;

		while ( q < layers ) {
#ifdef STATEFUL
			if ( q == 0 ) 
				lstm_cache_container_set_start(cache_layers[q][0]);
			else 
				lstm_next_state_copy(stateful_d_next[q], cache_layers[q][0], 0);
#else 
			lstm_cache_container_set_start(cache_layers[q][0]);
#endif
			++q;
		}

		unsigned int check = i % training_points;

		trailing = model->params->mini_batch_size;

		if ( i + model->params->mini_batch_size >= training_points ) {
			trailing = training_points - i;
		}

		q = 0;

		while ( q < trailing ) {
			e1 = q;
			e2 = q + 1;
			
			e3 = i % training_points;

			tmp_count = 0;
			while ( tmp_count < F ){
				first_layer_input[tmp_count] = 0.0; 
				++tmp_count;
			}

			first_layer_input[X_train[e3]] = 1.0;

			/* Layer numbering starts at the output point of the net */
			p = layers - 1;
			lstm_forward_propagate(model_layers[p], first_layer_input, cache_layers[p][e1], cache_layers[p][e2], p == 0);

			if ( p > 0 ) {
				--p;
				while ( p >= 0 ) {
					lstm_forward_propagate(model_layers[p], cache_layers[p+1][e2]->probs, cache_layers[p][e1], cache_layers[p][e2], p == 0);	
					--p;
				}
				p = 0;
			}

			loss_tmp += cross_entropy( cache_layers[p][e2]->probs, Y_train[e3]);
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

#ifdef STATEFUL
		p = 0;
		while ( p < layers ) {
			lstm_next_state_copy(stateful_d_next[p], cache_layers[p][e2], 1);
			++p;
		}
		p = 0;
#endif

		p = 0;
		while ( p < layers ) {
			lstm_zero_the_model(gradient_layers[p]);
			lstm_zero_d_next(d_next_layers[p], F);
			++p;
		}

		while ( q > 0 ) {
			e1 = q;
			e2 = q - 1;

			e3 = ( training_points + i - 1 ) % training_points;

			p = 0;
			while ( p < layers ) {
				lstm_zero_the_model(gradient_layers_entry[p]);
				++p;
			}


			p = 0;
			lstm_backward_propagate(model_layers[p], cache_layers[p][e1]->probs, Y_train[e3], d_next_layers[p], cache_layers[p][e1], gradient_layers_entry[0], d_next_layers[p]);

			if ( p < layers ) {
				++p;
				while ( p < layers ) {
					lstm_backward_propagate(model_layers[p], d_next_layers[p-1]->dldY_pass, -1, d_next_layers[p], cache_layers[p][e1], gradient_layers_entry[p], d_next_layers[p]);	
					++p;
				}
			}

			p = 0; 

			while ( p < layers ) {
				sum_gradients(gradient_layers[p], gradient_layers_entry[p]);
				++p;
			}

			i--; q--;
		}

		assert(check == e3);

		p = 0;
		while ( p < layers ) {

			if ( model->params->gradient_clip )
				gradients_clip(gradient_layers[p], model->params->gradient_clip_limit);

			if ( model->params->gradient_fit )
				gradients_fit(gradient_layers[p], model->params->gradient_clip_limit);

			++p;
		}

		p = 0;

		switch ( model->params->optimizer ) {
		case OPTIMIZE_ADAM:
			while ( p < layers ) {
				gradients_adam_optimizer(model_layers[p], gradient_layers[p], M_layers[p], R_layers[p], n);
				++p;
			}
		break;
		case OPTIMIZE_GRADIENT_DESCENT:
			while ( p < layers ) {
				gradients_decend(model_layers[p], gradient_layers[p]);
				++p;
			}
		break;
		default:
		break;
		}


		if ( !( n % PRINT_EVERY_X_ITERATIONS ) ) {

			status = 0;
			memset(time_buffer, '\0', sizeof time_buffer);
			time(&time_iter);
			strftime(time_buffer, sizeof time_buffer, "%X", localtime(&time_iter));

			printf("%s Iteration: %lu (epoch: %lu), Loss: %lf, record: %lf (iteration: %d), LR: %lf\n", time_buffer, n, epoch, loss, record_keeper, record_iteration, model->params->learning_rate);
			printf("=====================================================\n");

			lstm_output_string_layers(model_layers, char_index_mapping, X_train[b], NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING, layers);

			printf("\n=====================================================\n");
			
			// Flushing stdout
			fflush(stdout);
		}

		if ( !(n % STORE_PROGRESS_EVERY_X_ITERATIONS ))
			lstm_store_progress(n, loss);

		if ( b + model->params->mini_batch_size >= training_points )
			epoch++;

		i = (b + model->params->mini_batch_size) % training_points;

		if ( i < model->params->mini_batch_size){
			i = 0;
		}

#ifdef DECREASE_LR
		model->params->learning_rate = initial_learning_rate / ( 1.0 + n / model->params->learning_rate_decrease );
//		printf("learning rate: %lf\n", model->params->learning_rate);
#endif

		++n;
	}

	p = 0;
	while ( p < layers ) {
		lstm_values_next_cache_free(d_next_layers[p]);

		i = 0;
		while ( i < model->params->mini_batch_size) {
			lstm_cache_container_free(cache_layers[p][i]);
			lstm_cache_container_free(cache_layers[p][i]);
			++i;
		}

		lstm_free_model(M_layers[p]);
		lstm_free_model(R_layers[p]);

		lstm_free_model(gradient_layers_entry[p]);
		lstm_free_model(gradient_layers[p]);

		++p;
	}

#ifdef STATEFUL
	i = 0;
	while ( i < layers) {
		free(stateful_d_next[i]);
		++i;
	}
	free(stateful_d_next);
#endif

	free(cache_layers);
	free(gradient_layers);
	free(M_layers);
	free(R_layers);

}












