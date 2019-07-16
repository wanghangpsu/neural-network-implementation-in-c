//
//  model.c
//  
//
//  Created by hangwang on 10/6/18.
//

#include "model.h"

#define LEARNING_RATE 1
#define MAX_STEPS 100000
#define CONVERGENCE_CRITERION 0.001
#define LAYER1_INPUT_SIZE 8
#define LAYER2_INPUT_SIZE 8
#define LAYER3_INPUT_SIZE 8



#define INPUT_DEGREE 9
#define CLASS_NUM 7


int main()
{
    layer* layer1;
    layer* layer2;
    layer* layer3;
    layer* out_layer;
    
    
    dataset* data;
    dataset* test_data;
    double** error;
    double cost;
    int step;
    int  m , j;
    FILE* fileptr;
    
    char *filename = "train_norm.txt";
    char *test_filename = "test_norm.txt";
    char *save_dic = "cost.txt";
    data = load_data(filename);
    
    /* code for test y*/
//    for (m = 0; m < data->datasize; m++)
//    {
//        for(j = 0; j < CLASS_NUM; j++)
//        {
//            printf("%lf ", data->y[m][j]);
//        }
//        printf("\n");
//    }
    

    layer1 = initialize_layer(LAYER1_INPUT_SIZE, LAYER2_INPUT_SIZE, data->datasize);
    layer2 = initialize_layer(LAYER2_INPUT_SIZE, LAYER3_INPUT_SIZE,data->datasize);
    
    out_layer =initialize_layer(LAYER3_INPUT_SIZE, CLASS_NUM, data->datasize);

    /*connect the layer*/
    layer1->input = data->x;
    layer2->input = layer1->output;
    out_layer->input = layer2->output;


    layer_forward(layer1);
    layer_forward(layer2);
   
    layer_forward(out_layer);
    
    /*code for test the output*/
//    for (m = 0; m < data->datasize; m++)
//    {
//        for(j = 0; j < CLASS_NUM; j++)
//        {
//                printf("%lf ", out_layer->output[m][j]);
//        }
//        printf("\n");
//    }
    
    
//
    /*malloc error*/
    error = malloc(data->datasize * sizeof(double*));
    for (m = 0; m < data->datasize; m++)
    {
        error[m] = malloc(out_layer->output_size * sizeof(double));
    }


    error = compute_output_error(error, data->y, out_layer->output, data->datasize);
    /*cost before training*/
    cost = compute_cost(data->y, out_layer->output, data->datasize);
    printf("cost = %10lf\n", cost);
//
    /* training*/
    printf("begin to train\n");
    fileptr = fopen(save_dic, "w+");
    for (step = 0; step < MAX_STEPS; step++)
    {
        error = compute_output_error(error, data->y, out_layer->output, data->datasize);

        /* backward part*/
        back_propagate(error, out_layer);
        back_propagate(out_layer->input_error, layer2);
        back_propagate(layer2->input_error, layer1);

        /* forward part*/
        layer_forward(layer1);
        layer_forward(layer2);
        layer_forward(out_layer);



        cost = compute_cost(data->y, out_layer->output, data->datasize);
        fprintf(fileptr, "%lf\n",cost);
        //printf("in step %d, cost = %10lf\n", step, cost);
        if ((step % 100) == 0)
        {
            fprintf(fileptr, "%lf\n",cost);
            printf("in step %d, cost = %10lf\n", step, cost);
        }
    }
//    for (m = 0; m < data->datasize; m++)
//    {
//        printf("%10lf %10lf\n", out_layer->output[m][0], data->y[m]);
//    }
//
//    /*print the final weights and bias*/
//    printf("layer1 weights\n%10lf  %10lf  %10lf\n", layer1->weights[0][0], layer1->weights[1][0], layer1->bias[0]);
//    printf("%10lf  %10lf  %10lf\n", layer1->weights[0][1], layer1->weights[1][1], layer1->bias[1]);
//    printf("layer2 weights\n%10lf  %10lf  %10lf\n", out_layer->weights[0][0], out_layer->weights[1][0], out_layer->bias[0]);

    printf("train_accuracy = %lf\n", compute_accuracy(data->y, out_layer->output, data->datasize));

  
    
 /* compute test accuracy*/
    test_data = load_data(test_filename);
    layer1->input = test_data->x;

    /*layer forward*/
    layer_forward(layer1);
    layer_forward(layer2);
    layer_forward(out_layer);

    printf("test_accuracy = %lf\n", compute_accuracy(test_data->y, out_layer->output, test_data->datasize));
    
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
            ly->weights[i][j] = (rand() / (double)(RAND_MAX) -0.5) ; // random number in [-5,5]
            
            
        }
        ly->bias[j] = (rand() / (double)(RAND_MAX) - 0.5); // random number in [-5,5]
        
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

//double** compute_output_error(double** error, double** y, double** output, int datasize)
//{
//    int m, j;
//    for (m = 0; m < datasize; m++)
//    {
//        for (j = 0; j < CLASS_NUM; j++)
//        {
//            error[m][j] = y[m][j] - output[m][j];
//        }
//    }
//    return(error);
//}

double** compute_output_error(double** error, double** y, double** output, int datasize)
{
    int m, j;
    for (m = 0; m < datasize; m++)
    {
        for (j = 0; j < CLASS_NUM; j++)
        {
            error[m][j] = y[m][j] / output[m][j] + (y[m][j] - 1) / (1 - 0.99999999 * output[m][j]);
        }
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


double compute_cost(double** target_output, double** output, int datasize)
{
    int m, j;
    double cost;
    
    
    cost = 0.0;
    for (m = 0; m < datasize; m++)
    {
        for (j = 0; j < CLASS_NUM; j++)
        {
            
            cost += target_output[m][j] * log(output[m][j])  +  (1 - target_output[m][j]) * log(1 - 0.99999999 * output[m][j]);
        }
    }
    return(cost/datasize);
}

//double compute_cost(double** target_output, double** output, int datasize)
//{
//    int m, j;
//    double cost;
//
//
//    cost = 0.0;
//    for (m = 0; m < datasize; m++)
//    {
//        for (j = 0; j < CLASS_NUM; j++)
//        {
//
//        cost += 0.5 * (target_output[m][j] - output[m][j]) * (target_output[m][j] - output[m][j]);
//        }
//    }
//    return(cost);
//}

double compute_accuracy(double** target_output, double** output, int datasize)
{
    double accuracy;
    int m, true_label, j, class;
    double prob, max_prob;
    true_label = 0;
    
    for (m = 0; m < datasize; m++)
    {
        max_prob = 0;
        for(j = 0; j < CLASS_NUM; j++)
        {
            prob = output[m][j];
            if (prob > max_prob)
            {
                class = j;
                max_prob = prob;
            }
            
            
        }
        if (target_output[m][class] == 1)
        {
            true_label++;
        }
//        printf("data = %d  classlabel = %d  true? /%lf\n", m, class, target_output[m][class]);

    }
    
    accuracy = (double)true_label / datasize;
    
    return(accuracy);
}

dataset* load_data(char* data_filename)
{
    FILE* fileptr;
    int datasize, m, c;
    double input1, input2, input3, input4, input5, input6, input7, input8, input9, output;
    dataset* data;
    printf("reading data from %s...\n", data_filename);
    datasize = 0;
    fileptr = fopen(data_filename, "r");
    data = malloc(sizeof(dataset));
    
    while((fscanf(fileptr, "%10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf", &input1, &input2, &input3, &input4, &input5, &input6, &input7, &input8,&input9, &output) != EOF))
    {
        
        datasize++;
    }
    
    data->x = malloc(datasize * sizeof(double*));
    data->y = malloc(datasize * sizeof(double*));
    for (m = 0; m < datasize; m++)
    {
        data->x[m] = malloc(INPUT_DEGREE * sizeof(double));
        data->y[m] = malloc(CLASS_NUM * sizeof(double));
        for (c = 0; c < CLASS_NUM; c++)
        {
            data->y[m][c] = 0;
        }
    }
    rewind(fileptr);
    
    
    printf("the data size is: %d\n", datasize);
    m = 0;
    
    while((fscanf(fileptr, "%10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf %10lf", &input1, &input2, &input3, &input4, &input5, &input6, &input7, &input8, &input9, &output) != EOF))
    {
        
        data->x[m][0] = input1;
        data->x[m][1] = input2;
        data->x[m][2] = input3;
        data->x[m][3] = input4;
        data->x[m][4] = input5;
        data->x[m][5] = input6;
        data->x[m][6] = input7;
        data->x[m][7] = input8;
        data->x[m][8] = input9;
        data->y[m][((int)output - 1)] = 1;
        m ++;
        
    }
    data->datasize = datasize;
    
    return (data);
}
