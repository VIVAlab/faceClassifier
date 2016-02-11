#include "global.h"

float MultiplyByElement5_(float m1[][5], float m2[][5]);

float MultiplyByElement5(float m1[][5][5], float m2[][5][5], int outer_loop);

float MultiplyByElement18(float *m1, float m2[][18][18], int size);

float* CaliLayer48(float **img, int height, int width, int channels){
    int i, j, k, l;
    float img_segment[5][5];

    char path[50];
    strcpy(path, "");
    strcat(path, FILE_PATH);
    strcat(path, "weights/48cnet.bin");

    // read the weight and bias
    const int Depth = 64;
    const int Depth2 = 64;
    const int Depth3 = 256;
    const int Depth4 = 256;
    const int Filter = 25;
    const int Filter2 = 64*5*5;
    const int Filter3 = 64*18*18;
    const int Linear = 45;

    float *weight = malloc(Depth * Filter * sizeof(*weight));
    float *bias = malloc(Depth * sizeof(*bias));

    float *weight2 = malloc(Depth2 * Filter2 * sizeof(*weight2));
    float *bias2 = malloc(Depth2 * sizeof(*bias2));

    float *weight3 = malloc(Depth3 * Filter3 * sizeof(*weight3));
    float *bias3 = malloc(Depth3 * sizeof(*bias3));

    float *weight4 = malloc(Depth4 * Linear * sizeof(*weight4));
    float *bias4 = malloc(Depth4 * sizeof(*bias4));

    FILE *f = fopen(path, "rb");
    assert(fread(weight, sizeof(*weight), Depth*Filter, f) == Depth*Filter);
    assert(fread(bias, sizeof(*bias), Depth, f) == Depth);

    assert(fread(weight2, sizeof(*weight2), Depth2*Filter2, f) == Depth2*Filter2);
    assert(fread(bias2, sizeof(*bias2), Depth2, f) == Depth2);

    assert(fread(weight3, sizeof(*weight3), Depth3*Filter3, f) == Depth3*Filter3);
    assert(fread(bias3, sizeof(*bias3), Depth3, f) == Depth3);

    assert(fread(weight4, sizeof(*weight4), Depth4*Linear, f) == Depth4*Linear);
    assert(fread(bias4, sizeof(*bias4), Depth4, f) == Depth3);

    fclose(f);

    float filter[64][5][5];

    for (k = 0; k < 64; k++){
        for (i = 0; i < 5; i++){
            for (j = 0; j < 5; j++){
                filter[k][i][j] = weight[25*k + 5*i + j];
            }
        }
    }

    float filter2[64][64][5][5];
    // output the filter2
    for (l = 0; l < 64; l++){
        for (k = 0; k < 64; k++){
            for (i = 0; i < 5; i++){
                for (j = 0; j < 5; j++){
                    filter2[l][k][i][j] = weight2[64*5*5*l + 25*k + 5*i + j];
                }
            }
        }
    }

    float *filter3 = malloc(256*64*18*18*sizeof(float));
    for (l = 0; l < 256; l++){
        for (k = 0; k < 64; k++){
            for (i = 0; i < 18; i++){
                for (j = 0; j < 18; j++){
                    filter3[64*18*18*l + 18*18*k + 18*i + j] = weight3[64*18*18*l + 18*18*k + 18*i + j];
                    /*
                    if (k == 0 && i == 0 && j==0)
                        printf("filter3[%d][%d][%d][%d] = %f\n", l, k, i, j, filter3[64*18*18*l + 18*18*k + 18*i + j] = weight3[64*18*18*l + 18*18*k + 18*i + j]);
                    */
                }
            }
        }
    }


    float linear_para[45][256];

    // output the linear weights
    for (i = 0; i < 45; i++){
        for (j = 0; j < 256; j++){
            linear_para[i][j] = weight4[256*i + j];
            // printf("linear parameter[%d][%d] = %f\n", i, j, weight4[256*i + j]);
        }
    }

    float output1[64][44][44];

    // convolution
    int row, col, filter_num;
    float res;
    for (filter_num = 0; filter_num < 64; filter_num++){
        for (row = 0; row < 44; row++){
            for (col = 0; col < 44; col++){
                for (i = 0; i < 5; i++){
                    for (j = 0; j < 5; j++){
                        img_segment[i][j] = img[i+row][j+col];
                    }
                }

                res = MultiplyByElement5_(filter[filter_num], img_segment);
                res += bias[filter_num];

                output1[filter_num][row][col] = res;
                // printf("output1[%d][%d][%d] = %f\n", filter_num, row, col, res);
            }
        }
    }

    // padding
    float input2[64][46][46];
    for (k = 0; k < 64; k++){
        for (i = 0; i < 46; i++){
            for (j = 0; j < 46; j++){
                if (i == 0 || i == 45 || j == 0 || j == 45){
                    input2[k][i][j] = 0;
                } else {
                    input2[k][i][j] = output1[k][i-1][j-1];
                }
            }
        }
    }

    // polling and RELU
    float output2[64][22][22];
    int stride2 = 2;
    float maxpool;
    for (k = 0; k < 64; k++){
        for (row = 0; row < 22; row++){
            for (col = 0; col < 22; col++){
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

    float output3[64][18][18];
    float segment[64][5][5];
    // convolution2
    for (filter_num = 0; filter_num < 64; filter_num++){
        for (row = 0; row < 18; row++){
            for (col = 0; col < 18; col++){
                for(k = 0; k < 64; k++){
                    for (i = 0; i < 5; i++){
                        for (j = 0; j < 5; j++){
                            segment[k][i][j] = output2[k][i+row][j+col];
                        }
                    }
                }

                res = MultiplyByElement5(filter2[filter_num], segment, 64);
                res += bias2[filter_num];

                output3[filter_num][row][col] = res;
                // printf("output3[%d][%d][%d] = %f\n", filter_num, row, col, res);
            }
        }
    }

    /*
    // padding
    float input4[64][20][20];
    for (k = 0; k < 64; k++){
        for (i = 0; i < 20; i++){
            for (j = 0; j < 20; j++){
                if (i == 0 || i == 19 || j == 0 || j == 19){
                    input4[k][i][j] = 0;
                } else {
                    input4[k][i][j] = output3[k][i-1][j-1];
                }
            }
        }
    }

    // polling and RELU
    float output5[64][9][9];
    stride2 = 2;
    for (k = 0; k < 64; k++){
        for (row = 0; row < 9; row++){
            for (col = 0; col < 9; col++){
                maxpool = 0.0;
                for (i = 0; i < 3; i++){
                    for (j = 0; j < 3;j++){
                        maxpool = max(maxpool, input4[k][i+row*stride2][j+col*stride2]);
                        // printf("maxpool: %f, input[%d][%d]: %f\n", maxpool, i + row*2, j + col*2, input2[k][i+row*stride2][j+col*stride2]);
                    }
                }
                output5[k][row][col] = maxpool;
                // printf("result: %f, filter: %d, row: %d, col: %d\n", maxpool, k, row, col);
            }
        }
    }
    */

    // RELU
    for (k = 0; k < 64; k++){
        for (row = 0; row < 18; row++){
            for (col = 0; col < 18; col++){
                if (output3[k][row][col] < 0)
                    output3[k][row][col] = 0;
            }
        }
    }

    // convolution 3 & RELU
    float output6[256];
    for (i = 0; i < 256; i++){
        output6[i] = bias3[i] + MultiplyByElement18(&filter3[i*64*18*18], output3, 64);
        if (output6[i] < 0){
            output6[i] = 0.0;
        }
        // printf("output6[%d] = %f\n", i, output6[i]);
    }

    /*
    // RELU
    for (i = 0; i < 256; i++){
        if (output6[i] < 0)
            output6[i] = 0;
    }
    */


    // linear
    float output7[45];
    for (i = 0; i < 45; i++){
        output7[i] = 0;
        for (j = 0; j < 256; j++){
            output7[i] += output6[j] * linear_para[i][j];
        }
        output7[i] += bias4[i];
        // printf("output7[%d] = %f\n", i, output7[i]);
    }

    // softmax
    float *out_48c = malloc(sizeof(*out_48c));
    float out[45];
    float minimum = output7[0];
    float sum = 0;
    float logsum;
    float detect[45][3];


    for (i = 0; i < 45; i++){
        minimum = min(output7[i], minimum);
    }
    for (i = 0; i < 45; i++){
        sum += expf(output7[i] - minimum);
    }
    logsum = minimum + logf(sum);

    // different scales
    float s[5] = {0.83, 0.91, 1.0, 1.10, 1.21};
    float x[3] = {-0.17, 0.0, 0.17};
    float y[3] = {-0.17, 0.0, 0.17};
    float sn = 0;
    float xn = 0;
    float yn = 0;

    // the threshold
    float thres = 0.0;

    int m = 0;
    for (i = 0; i < 5; i++){
        for (j = 0; j < 3; j++){
            for (k = 0; k < 3; k++){
                // printf("output7 - logsum = %f\n", output7[9*i+3*j+k] - logsum);
                out[9*i+3*j+k] = expf(output7[9*i+3*j+k] - logsum);
                // printf("out[%d] = %f\n", 9*i+3*j+k, out[9*i+3*j+k]);
                if (out[9*i+3*j+k] > thres){
                    // printf("out[%d] = %f\n", 9*i+3*j+k, out[9*i+3*j+k]);
                    // printf("i: %d, j: %d, k: %d\n", i, j, k);

                    sn += s[i];
                    xn += x[j];
                    yn += y[k];
                    m++;
                }
            }
        }
    }

    sn /= m;
    xn /= m;
    yn /= m;

    out_48c[0] = sn;
    out_48c[1] = xn;
    out_48c[2] = yn;

    return out_48c;
    /*
    printf("output 1: %f\n", output8[0]);
    printf("output 2: %f\n", output8[1]);
    printf("out: %f\n", out[0]);
    printf("out: %f\n", out[1]);
    */


    /*
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

}
