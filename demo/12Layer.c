#include "global.h"

float MultiplyByElement3(float m1[][3], float m2[][3]);

float MultiplyByElement5(float m1[][5][5], float m2[][5][5], int outer_loop);

float Layer12(float **img, int height, int width, int channels){
    int i, j, k, l;
    float img_segment[3][3];
    float filter[16][3][3];
    float filter2[16][16][5][5];
    float output1[16][10][10];

    //char path[] = "/home/binghao/cnn/12net/module1.bin";
    // char path[] = "/Users/wbh/cnn/12net/module1.bin";
    char path[50];
    strcpy(path, "");
    strcat(path, FILE_PATH);
    strcat(path, "weights/12net.bin");

    // read the weight and bias
    const int Depth = 16;
    const int Filter = 9;
    const int Filter2 = 400;

    float *weight = malloc(Depth * Filter * sizeof(*weight));
    float *bias = malloc(Depth * sizeof(*bias));

    float *weight2 = malloc(Depth * Filter2 * sizeof(*weight2));
    float *bias2 = malloc(Depth * sizeof(*bias2));

    float *weight3 = malloc(Depth * 2 * sizeof(*weight3));
    float *bias3 = malloc(Depth * sizeof(*bias3));

    FILE *f = fopen(path, "rb");
    assert(fread(weight, sizeof(*weight), Depth*Filter, f) == Depth*Filter );
    assert(fread(bias, sizeof(*bias), Depth, f) == Depth);

    assert(fread(weight2, sizeof(*weight2), Depth*Filter2, f) == Depth*Filter2);
    assert(fread(bias2, sizeof(*bias2), Depth, f) == Depth);

    assert(fread(weight3, sizeof(*weight3), Depth*2, f) == Depth*2);
    assert(fread(bias3, sizeof(*bias3), Depth, f) == Depth);

    fclose(f);

    // output the filter
    for (k = 0; k < 16; k++){
        for (i = 0; i < 3; i++){
            for (j = 0; j < 3; j++){
                // printf("filter[%d][%d][%d] =  %f\n", k, i, j, weight[9*k + 3*i + j]);
                filter[k][i][j] = weight[9*k + 3*i + j];
            }
        }
    }

    // output the filter2
    for (l = 0; l < 16; l++){
        for (k = 0; k < 16; k++){
            for (i = 0; i < 5; i++){
                for (j = 0; j < 5; j++){
                    // printf("filter2[%d][%d][%d][%d] =  %f\n", l, k, i, j, weight2[400*l + 25*k + 5*i + j]);
                    filter2[l][k][i][j] = weight2[400*l + 25*k + 5*i + j];
                }
            }
        }
    }

    float linear_para[2][16];
    // output the linear weights
    for (i = 0; i < 2; i++){
        for (j = 0; j < 16; j++){
            linear_para[i][j] = weight3[16*i + j];
            // printf("linear parameter[%d][%d] = %f\n", i, j, weight3[16*i + j]);
        }
    }

    // convolution
    int row, col, filter_num;
    float res;
    for (filter_num = 0; filter_num < 16; filter_num++){
        for (row = 0; row < 10; row++){
            for (col = 0; col < 10; col++){
                for (i = 0; i < 3; i++){
                    for (j = 0; j < 3; j++){
                        img_segment[i][j] = img[i+row][j+col];
                    }
                }

                res = MultiplyByElement3(filter[filter_num], img_segment);
                res += bias[filter_num];

                output1[filter_num][row][col] = res;
                // printf("output1[%d][%d][%d] = %f\n", filter_num, row, col, res);
            }
        }
    }


    // padding
    float input2[16][12][12];
    for (k = 0; k < 16; k++){
        for (i = 0; i < 12; i++){
            for (j = 0; j < 12; j++){
                if (i == 0 || i == 11 || j == 0 || j == 11){
                    input2[k][i][j] = 0;
                } else {
                    input2[k][i][j] = output1[k][i-1][j-1];
                }
            }
        }
    }

    // polling and RELU
    float output2[16][5][5];
    int stride2 = 2;
    float maxpool;
    for (k = 0; k < 16; k++){
        for (row = 0; row < 5; row++){
            for (col = 0; col < 5; col++){
                maxpool = 0.0;
                for (i = 0; i < 3; i++){
                    for (j = 0; j < 3;j++){
                        maxpool = max(maxpool, input2[k][i+row*stride2][j+col*stride2]);
                        // printf("maxpool: %f, input[%d][%d]: %f\n", maxpool, i + row*2, j + col*2, input2[k][i+row*stride2][j+col*stride2]);
                    }
                }
                output2[k][row][col] = maxpool;
                // printf("result: %f, filter: %d, row: %d, col: %d\n", maxpool, k, row, col);
            }
        }
    }

    // convolution 2
    float output4[16];
    for (i = 0; i < 16; i++){
        output4[i] = bias2[i] + MultiplyByElement5(filter2[i], output2, 16);
        if (output4[i] < 0){
            output4[i] = 0.0;
        }
        // printf("output4[%d] = %f\n", i, output4[i]);
    }

    // linear
    float output7[2];
    for (i = 0; i < 2; i++){
        output7[i] = 0;
        for (j = 0; j < 16; j++){
            output7[i] += output4[j] * linear_para[i][j];
        }
        output7[i] += bias3[i];
        // printf("output7[%d] = %f\n", i, output7[i]);
    }

    // softmax
    float output8[2];
    float out[2], logsum;
    float a, b, c, d;
    a = output7[0];
    b = output7[1];
    c = a < b ? a : b;      // c = min(a, b)
    d = a + b - c;          // d = max(a, b)
    logsum = c + logf(1 + expf(d - c));
    output8[0] = a - logsum;
    output8[1] = b - logsum;
    out[0] = expf(output8[0]);
    out[1] = expf(output8[1]);

    /* write results into a file
    FILE *ffp = fopen(pooling_output_path,"w");
    for (k = 0; k < 16; k++){
        for (i = 0; i < 5; i++){
            for (j = 0; j < 5; j++){
                fprintf(ffp, "output2[%d][%d][%d]: %f\n", k, i, j, output2[k][i][j]);
            }
        }
        fprintf(ffp, "---------------------------------\n");
    }

    fclose(ffp);

    FILE *fp = fopen(conv_layer_output_path,"w");
    if (f == NULL){
        printf("Error opening file!\n");
        exit(1);
    }

    for (filter_num = 0; filter_num < 16; filter_num++){
        for (row = 0; row < 10; row++){
            // printf("-----------------------\n");
            fprintf(fp, "-----------------------\n");
            for (col = 0; col < 10; col++){
                // printf("value of filter_num %d at row %d, col %d is : %f\n", filter_num, row, col, output[filter_num][row][col]);
                fprintf(fp, "value of filter_num %d at row %d, col %d is : %f\n", filter_num, row, col, output1[filter_num][row][col]);
            }
        }
    }
    printf("output size: %d * %d *%d", row, col, filter_num);

    fclose(fp);

    */

    return out[0];
}
