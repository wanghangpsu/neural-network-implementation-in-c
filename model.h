//
//  model.h
//  
//
//  Created by 王航 on 10/6/18.
//

#ifndef model_h
#define model_h

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "mlp.h"



layer* initialize_layer(int input_size, int output_size, int datasize);
void free_layer(layer* );
double sigmoid(double x);
void layer_forward(layer* );
double** compute_output_error(double** error, double* y, double** output, int datasize);
void back_propagate(double** error, layer* ly);
double compute_cost(double* target_output, double** output, int datasize);
dataset* load_data(char* data_filename);
#endif /* model_h */
