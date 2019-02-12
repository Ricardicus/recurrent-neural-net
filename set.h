#ifndef LSTM_SET_H
#define LSTM_SET_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>

#define	SET_MAX_CHARS	1000

typedef struct set_T {
	char values[SET_MAX_CHARS];
	int free[SET_MAX_CHARS];
} set_T;

int set_insert_symbol(set_T*, char);
char set_indx_to_char(set_T*, int);
int set_char_to_indx(set_T*, char);
int set_probability_choice(set_T*, double*);
int set_greedy_argmax(set_T*, double*);
int set_get_features(set_T*);

void set_print(set_T*, double*);

void initialize_set(set_T*);

void set_store_as_json(set_T *, FILE*);
void set_store(set_T *, FILE*);
int set_read(set_T *, FILE*);

#endif
