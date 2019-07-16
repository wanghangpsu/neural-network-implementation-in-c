//
//  mlp.h
//  
//
//  Created by hangwang on 10/6/18.
//

#ifndef mlp_h
#define mlp_h

typedef struct
{
    int datasize;
    double** input;
    int input_size;
    double** output;
    int output_size;
    double** input_error;
    double** weights;
    double* bias;
} layer;

typedef struct
{
    double** x;
    double* y;
    int datasize;
} dataset;
#endif /* mlp_h */
