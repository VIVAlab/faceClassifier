#include "global.h"

float MultiplyByElement5_(float m1[][5], float m2[][5]);

float MultiplyByElement10(float m1[][10][10], float m2[][10][10], int outer_loop);

float* CaliLayer24(float **img, int height, int width, int channels){
    int i, j, k, l;
    float img_segment[5][5];
    float filter[32][5][5];
    float filter2[64][32][10][10];
    float output1[32][20][20];

    //char path[] = "/home/binghao/cnn/24cnet/24cnet.bin";
    // char path[] = "/Users/wbh/cnn/24cnet/24cnet.bin";
    char path[250];
    strcpy(path, "");
    strcat(path, FILE_PATH);
    strcat(path, "weights/24cnet.bin");


    // char conv_layer_output_path[] = "/home/binghao/cnn/conv_layer_output.txt";
    // char conv_layer_output_path[] = "/Users/wbh/cnn/conv_layer_output.txt";

    // char pooling_output_path[] = "/home/binghao/cnn/pooling_ouput.txt";
    // char pooling_output_path[] = "/Users/wbh/cnn/pooling_ouput.txt";

    /*
    // output the image data
    printf("image data start");
    for (i = 0; i < height; i++){
        for (j = 0; j < width; j++){
            printf("img[%d][%d] = %f\n", i, j, img[i][j]);
        }
    }
    printf("image data end");
    */

    // read the weight and bias
    const int Depth = 32;
    const int Depth2 = 64;
    const int Filter = 25;
    const int Filter2 = 3200;
    const int Linear = 45;

    float *weight = malloc(Depth * Filter * sizeof(*weight));
    float *bias = malloc(Depth * sizeof(*bias));

    float *weight2 = malloc(Depth2 * Filter2 * sizeof(*weight2));
    float *bias2 = malloc(Depth2 * sizeof(*bias2));

    float *weight3 = malloc(Depth2 * Linear * sizeof(*weight3));
    float *bias3 = malloc(Depth2 * sizeof(*bias3));

    FILE *f = fopen(path, "rb");
    assert(fread(weight, sizeof(*weight), Depth*Filter, f) == Depth*Filter );
    assert(fread(bias, sizeof(*bias), Depth, f) == Depth);

    assert(fread(weight2, sizeof(*weight2), Depth2*Filter2, f) == Depth2*Filter2);
    assert(fread(bias2, sizeof(*bias2), Depth2, f) == Depth2);

    assert(fread(weight3, sizeof(*weight3), Depth2*Linear, f) == Depth2*Linear);
    assert(fread(bias3, sizeof(*bias3), Depth2, f) == Depth2);

    fclose(f);


    /*
    // output the weight and bias of module4
    for (i = 0; i < Depth; i++){
        for (j = 0; j < Filter2; j++){
            printf("weight[%d, %d] = %f\n", i, j, weight2[Filter2 * i + j]);
        }
        printf("bias[%d] = %f\n", i, bias2[i]);
    }
    exit(0);
    */

    // output the filter
    for (k = 0; k < 32; k++){
        for (i = 0; i < 5; i++){
            for (j = 0; j < 5; j++){
                // printf("filter[%d][%d][%d] =  %f\n", k, i, j, weight[25*k + 5*i + j]);
                filter[k][i][j] = weight[25*k + 5*i + j];
            }
        }
    }


    // output the filter2
    for (l = 0; l < 64; l++){
        for (k = 0; k < 32; k++){
            for (i = 0; i < 10; i++){
                for (j = 0; j < 10; j++){
                    // printf("filter2[%d][%d][%d][%d] =  %f\n", l, k, i, j, weight2[6400*l + 100*k + 10*i + j]);
                    filter2[l][k][i][j] = weight2[3200*l + 100*k + 10*i + j];
                }
            }
        }
    }

    float linear_para[45][64];
    // output the linear weights
    for (i = 0; i < 45; i++){
        for (j = 0; j < 64; j++){
            linear_para[i][j] = weight3[64*i + j];
            // printf("linear parameter[%d][%d] = %f\n", i, j, weight3[64*i + j]);
        }
    }

    // convolution
    int row, col, filter_num;
    float res;
    for (filter_num = 0; filter_num < 32; filter_num++){
        for (row = 0; row < 20; row++){
            for (col = 0; col < 20; col++){
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
    float input2[32][22][22];
    for (k = 0; k < 32; k++){
        for (i = 0; i < 22; i++){
            for (j = 0; j < 22; j++){
                if (i == 0 || i == 21 || j == 0 || j == 21){
                    input2[k][i][j] = 0;
                } else {
                    input2[k][i][j] = output1[k][i-1][j-1];
                }
            }
        }
    }

    // polling and RELU
    float output2[32][10][10];
    int stride2 = 2;
    float maxpool;
    for (k = 0; k < 32; k++){
        for (row = 0; row < 10; row++){
            for (col = 0; col < 10; col++){
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
    float output4[64];
    for (i = 0; i < 64; i++){
        output4[i] = bias2[i] + MultiplyByElement10(filter2[i], output2, 32);
        if (output4[i] < 0){
            output4[i] = 0.0;
        }
        // printf("output4[%d] = %f\n", i, output4[i]);
    }

    // linear
    float output7[45];
    for (i = 0; i < 45; i++){
        output7[i] = 0;
        for (j = 0; j < 64; j++){
            output7[i] += output4[j] * linear_para[i][j];
        }
        output7[i] += bias3[i];
        // printf("output7[%d] = %f\n", i, output7[i]);
    }


    // softmax
    float *out_24c = malloc(sizeof(*out_24c));
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
    float thres = 0.1;

    int m = 0;
    for (i = 0; i < 5; i++){
        for (j = 0; j < 3; j++){
            for (k = 0; k < 3; k++){
                out[9*i+3*j+k] = expf(output7[9*i+3*j+k] - logsum);
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

    out_24c[0] = sn;
    out_24c[1] = xn;
    out_24c[2] = yn;

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

    return out_24c;
}
