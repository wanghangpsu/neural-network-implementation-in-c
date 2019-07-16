//
//  model.c
//  
//
//  Created by hangwang on 10/6/18.
//

#include "model.h"

#define LEARNING_RATE 1
#define MAX_STEPS 10000
#define CONVERGENCE_CRITERION 0.001
#define LAYER1_INPUT_SIZE 8
#define LAYER2_INPUT_SIZE 32
#define LAYER2_OUTPUT_SIZE 1
#define LAYER3_OUTPUT_SIZE 2
#define OUTPUT_SIZE 1


#define INPUT_DEGREE 384


int main()
{
    layer* layer1;
    layer* layer2;
    dataset* data;
    double** error;
    double cost;
    int step;
    int  m;
    FILE* fileptr;
    
    char *filename = "train_norm.txt";
    char *save_dic = "cost.txt";
    data = load_data(filename);
    /* PLM structure: two layers, two neurans at the hidden layer*/
    layer1 = initialize_layer(LAYER1_INPUT_SIZE, LAYER2_INPUT_SIZE, data->datasize);
    layer2 = initialize_layer(LAYER2_INPUT_SIZE, LAYER2_OUTPUT_SIZE, data->datasize);
    
    /*connect the layer*/
    layer1->input = data->x;
    layer2->input = layer1->output;
    
    
    layer_forward(layer1);
    layer_forward(layer2);
    
    /*malloc error*/
    error = malloc(data->datasize * sizeof(double*));
    for (m = 0; m < data->datasize; m++)
    {
        error[m] = malloc(layer2->output_size * sizeof(double));
    }
    
    
    error = compute_output_error(error, data->y, layer2->output, data->datasize);
    
    
    cost = compute_cost(data->y, layer2->output, data->datasize);
    
    
    //    printf("error\n%10lf  %10lf %10lf %10lf\n", error[0][0],error[1][0],error[2][0],error[3][0]);
    //    printf("layer1 weights\n%10lf  %10lf  %10lf\n", layer1->weights[0][0], layer1->weights[1][0], layer1->bias[0]);
    //    printf("%10lf  %10lf  %10lf\n", layer1->weights[0][1], layer1->weights[1][1], layer1->bias[1]);
    //    printf("layer2 weights\n%10lf  %10lf  %10lf\n", layer2->weights[0][0], layer1->weights[1][0], layer1->bias[0]);
    //    printf("%10lf  %10lf %10lf %10lf", layer2->output[0][0], layer2->output[1][0], layer2->output[2][0],layer2->output[3][0]);
    
    printf("cost = %10lf\n", cost);
    
    /* training*/
    fileptr = fopen(save_dic, "w+");
    for (step = 0; step < MAX_STEPS; step++)
    {
        error = compute_output_error(error, data->y, layer2->output, data->datasize);
        back_propagate(error, layer2);
        back_propagate(layer2->input_error, layer1);
        layer_forward(layer1);
        layer_forward(layer2);
        cost = compute_cost(data->y, layer2->output, data->datasize);
        fprintf(fileptr, "%lf\n",cost);
        printf("in step %d, cost = %10lf\n", step, cost);
    }
    for (m = 0; m < data->datasize; m++)
    {
        printf("%10lf %10lf\n", layer2->output[m][0], data->y[m]);
    }
    
   
    printf("layer2 weights\n%10lf  %10lf  %10lf\n", layer2->weights[0][0], layer2->weights[1][0], layer2->bias[0]);
    
}

layer* initialize_layer(int input_size, int output_size, int datasize)
{
    int i, j, m;
    layer* ly;
    ly = malloc(sizeof(layer));
   
    ly->datasize = datasize;
    ly->input_size = input_size;
    ly->output_size = output_size;
    
    ly->input = malloc(datasize * sizeof(double*));
    
    ly->output = malloc(datasize * sizeof(double*));
   
    ly->input_error = malloc(datasize * sizeof(double*));
    
    ly->weights = malloc(input_size * sizeof(double*));
    ly->bias = malloc(output_size * sizeof(double));
    for (i = 0; i < input_size; i++)
    {
        ly->weights[i] = malloc(output_size * sizeof(double));
    }
    for (m = 0; m < datasize; m++)
    {
        ly->input[m] = malloc(input_size * sizeof(double));
        ly->input_error[m] = malloc(input_size * sizeof(double));
        ly->output[m] = malloc(output_size * sizeof(double));
    }
    srand(1);
    for (j = 0; j < output_size; j++)
    {
     
        for (i = 0; i < input_size; i++)
        {
            ly->weights[i][j] = (rand() / (double)(RAND_MAX) -0.5)*5; // random number in [-5,5]
            
            
        }
        ly->bias[j] = (rand() / (double)(RAND_MAX) - 0.5)*5; // random number in [-5,5]
        
    }
    
    return(ly);
}

void free_layer(layer* ly)
{
    int j, m;
    free(ly->input);
    free(ly->output);
    free(ly->bias);
    
    for (j = 0; j < ly->output_size; j++)
    {
        free(ly->weights[j]);
    }
    for (m = 0; m < ly->datasize; m++)
    {
        free(ly->input[m]);
        free(ly->input_error[m]);
        free(ly->output[m]);
    }
    free(ly->weights);
    free(ly->input);
    free(ly->output);
    free(ly->input_error);
}

double sigmoid(double x)
{
    return(1 / (1 + exp(-x)));
}

void layer_forward(layer* ly)
{
    double y[ly->datasize][ly->output_size];
    int m, i, j;

    for (m = 0; m < ly->datasize; m++)
    {
        for (j = 0; j < ly->output_size; j++)
        {
            y[m][j] = 0;
            for (i = 0; i < ly->input_size; i++)
            {
                y[m][j] += ly->input[m][i] * ly->weights[i][j];
            }
            y[m][j] += ly->bias[j];
            //printf("%10lf", y[m][j]);
            
            //printf("%10lf= %10lf * %10lf + %10lf * %10lf + %10lf\n", y[m][j],ly->input[m][0],ly->weights[0][j], ly->input[m][1],ly->weights[m][1],ly->bias[j]);
        }
        //printf("\n");
        for (j = 0; j < ly->output_size; j++)
        {
            ly->output[m][j] = sigmoid(y[m][j]);
            //printf("%10lf", ly->output[m][j]);
            
        }
       // printf("\n");
        
    }
}

double** compute_output_error(double** error, double* y, double** output, int datasize)
{
    int m;
    for (m = 0; m < datasize; m++)
    {
        
            error[m][0] = y[m] - output[m][0];

    }
    return(error);
}


void back_propagate(double** error, layer* ly)
{
    // return the error of the last layer
    int i, j, m;
    double s;
    for (i = 0; i < ly->input_size; i++)
    {
        for(m = 0; m < ly->datasize; m++)
        {
            ly->input_error[m][i] = 0;
            for (j = 0; j < ly->output_size; j++)
            {
                ly->input_error[m][i] += error[m][j] * ly->weights[i][j] * (1 - ly->output[m][j]) * ly->output[m][j];
            }
        }
    }
    
    for (j = 0; j < ly->output_size; j++)
    {
        for (i = 0; i < ly->input_size; i++)
        {
            s = 0;
            for (m = 0; m < ly->datasize; m++)
            {
                s += error[m][j] *  (1 - ly->output[m][j]) * ly->output[m][j] * ly->input[m][i];
                
                //   ly->weights[i][j] += LEARNING_RATE * error[j] * ly->output[j] * (1 - ly->output[j]) * ly->input[i];
            }
            ly->weights[i][j] += LEARNING_RATE * s /ly->datasize;
        }
        s = 0;
        for (m = 0; m < ly->datasize; m++)
        {
            s += error[m][j] * ly->output[m][j] * (1 - ly->output[m][j]);
            //ly->bias[j] += LEARNING_RATE * error[j] * ly->output[j] * (1 - ly->output[j])
        }
        ly->bias[j] += LEARNING_RATE * s / ly->datasize;
    }
    
}


double compute_cost(double* target_output, double** output, int datasize)
{
    int m;
    double cost;
    
   
    cost = 0.0;
    for (m = 0; m < datasize; m++)
    {
        cost += 0.5 * (target_output[m] - output[m][0]) * (target_output[m] - output[m][0]);
    }
    return(cost);
}

dataset* load_data(char* data_filename)
{
    FILE* fileptr;
    int datasize, m;
    double input1, input2, input3, input4, input5, input6, input7, input8, output;
    dataset* data;
    printf("reading data from %s...\n", data_filename);
    datasize = 0;
    fileptr = fopen(data_filename, "r");
    data = malloc(sizeof(dataset));
    while((fscanf(fileptr, "%10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf", &input1, &input2, &input3, &input4, &input5, &input6, &input7, &input8, &output) != EOF))
    {
        
        datasize++;
    }
    
    
   
    
    
    data->x = malloc(datasize * sizeof(double*));
    data->y = (double*)malloc(datasize * sizeof(double));
    for (m = 0; m < datasize; m++)
    {
        data->x[m] = malloc(INPUT_DEGREE * sizeof(double));
    }
    rewind(fileptr);
    
    printf("the data size is: %d\n", datasize);
    m = 0;
    
    while((fscanf(fileptr, "%10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf", &input1, &input2, &input3, &input4, &input5, &input6, &input7, &input8, &output) != EOF))
    {
        
        data->x[m][0] = input1;
        data->x[m][1] = input2;
        data->x[m][2] = input3;
        data->x[m][3] = input4;
        data->x[m][4] = input5;
        data->x[m][5] = input6;
        data->x[m][6] = input7;
        data->x[m][7] = input8;
        data->y[m] = output;
        m ++;
        
       
        
    }
    data->datasize = datasize;
    
    return (data);
}
