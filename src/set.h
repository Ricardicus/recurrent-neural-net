/*
* This file is part of the LSTM Network implementation In C made by Rickard Hallerbäck
* 
*                 Copyright (c) 2018 Rickard Hallerbäck
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), 
* to deal in the Software without restriction, including without limitation the rights 
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
* Software, and to permit persons to whom the Software is furnished to do so, subject to 
* the following conditions:
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
*
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
* OR OTHER DEALINGS IN THE SOFTWARE.
*/
#ifndef LSTM_SET_H
#define LSTM_SET_H

/*! \file set.h
    \brief LSTM feature-to-index mapping
    
    Features get mapped to an index value.
    This process is done using the following definitions and functions.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>

#define	SET_MAX_CHARS	1000

typedef struct set_t {
  char values[SET_MAX_CHARS];
  int free[SET_MAX_CHARS];
} set_t;

int set_insert_symbol(set_t*, char);
char set_indx_to_char(set_t*, int);
int set_char_to_indx(set_t*, char);
int set_probability_choice(set_t*, double*);
int set_greedy_argmax(set_t*, double*);
int set_get_features(set_t*);

void set_print(set_t*, double*);

void initialize_set(set_t*);

void set_store_as_json(set_t *, FILE*);
void set_store(set_t *, FILE*);
int set_read(set_t *, FILE*);

#endif
