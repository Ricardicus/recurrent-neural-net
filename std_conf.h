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

/*
* Set the standard behaviour of the network
*/

/*! \file std_conf.h
    \brief Definitions that set the standard behaviour of the network
*/

#ifndef STD_CONF_H
#define STD_CONF_H

#define NEURONS                                                 68

#define STD_LEARNING_RATE                                       0.001
#define STD_MOMENTUM                                            0.0
#define STD_LAMBDA                                              0.05
#define SOFTMAX_TEMP                                            1.0
#define GRADIENT_CLIP_LIMIT                                     5.0
#define MINI_BATCH_SIZE                                         100
#define LOSS_MOVING_AVG                                         0.01

#define LAYERS                                                  3 // Has a tremendous impact on avaiable memory

#define STATEFUL                                                1

#define GRADIENTS_CLIP                                          1
#define GRADIENTS_FIT                                           0

#define MODEL_REGULARIZE                                        0

#define DECREASE_LR                                             0 // set to 0 to disable decreasing learning rate

#define STD_LEARNING_RATE_DECREASE                              100000

/*
* These defines modify how the program interacts with the user
* of the program during the training phase. Here you can
* decide how often it should output its progress and wether or not
* it should write to file among other things.
*/
#define PRINT_EVERY_X_ITERATIONS                                100
#define STORE_EVERY_X_ITERATIONS                                8000
#define PRINT_PROGRESS                                          1   // set to 0 to disable printing
#define PRINT_SAMPLE_OUTPUT                                     1   // set to 0 to disable output sampling
#define PRINT_SAMPLE_OUTPUT_TO_FILE                             0   // set to 0 to disable output sampling to file
#define PRINT_SAMPLE_OUTPUT_TO_FILE_ARG                         "a" // used as an argument to fopen (goes with "w" or "a")
#define PRINT_SAMPLE_OUTPUT_TO_FILE_NAME                        "progress_output.txt" // name of the file containing samples
#define STORE_PROGRESS_EVERY_X_ITERATIONS                       1000 // set to 0 to disable writing loss value to file during training
#define PROGRESS_FILE_NAME                                      "progress.csv"
#define NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING              200

/*
* Once the network has been trained it is stored to these files.
* The .net extension is not to be confused with microsoft,
* it is just an extension I picked that alludes to it being
* a 'network' in its rawest form. It can be parsed by the program
* but not by the interactive HTML application.
* For that, the .json file is intended to be used.
*/
#define STD_LOADABLE_NET_NAME                                   "lstm_net.net"
#define STD_JSON_NET_NAME                                       "lstm_net.json"

// ================== DO NOT CHANGE THE FOLLOWING DEFINES ======================
// Don't change this one, else the HTML application will not work.
#define JSON_KEY_NAME_SET                                       "Feature mapping"
// This define should be undeffed, it was defined during experimentation
// #define INTERLAYER_SIGMOID_ACTIVATION
// =============================================================================

#endif


