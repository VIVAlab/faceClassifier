#include "global.h"

float MultiplyByElement5_(float m1[][5], float m2[][5]);
float MultiplyByElement10(float m1[][10][10], float m2[][10][10], int outer_loop);

float Layer24(float **img, int height, int width, int channels, char *path){
    int i, j, k, l;
    float img_segment[5][5];
    float filter[64][5][5];
    float filter2[128][64][10][10];
    float output1[64][20][20];

    // char path[] = "/home/binghao/cnn/24net/24net.bin";
    // char path[] = "/Users/wbh/cnn/24net/24net.bin";
    // char path[250];
//     strcpy(path, "");
//     strcat(path, FILE_PATH);
//     strcat(path, "weights/24net.bin");



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
    const int Depth = 64;
    const int Depth2 = 128;
    const int Filter = 25;
    const int Filter2 = 6400;
    const int Linear = 2;

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
    for (k = 0; k < 64; k++){
        for (i = 0; i < 5; i++){
            for (j = 0; j < 5; j++){
                // printf("filter[%d][%d][%d] =  %f\n", k, i, j, weight[25*k + 5*i + j]);
                filter[k][i][j] = weight[25*k + 5*i + j];
            }
        }
    }


    // output the filter2
    for (l = 0; l < 128; l++){
        for (k = 0; k < 64; k++){
            for (i = 0; i < 10; i++){
                for (j = 0; j < 10; j++){
                    // printf("filter2[%d][%d][%d][%d] =  %f\n", l, k, i, j, weight2[6400*l + 100*k + 10*i + j]);
                    filter2[l][k][i][j] = weight2[6400*l + 100*k + 10*i + j];
                }
            }
        }
    }

    float linear_para[2][128];
    // output the linear weights
    for (i = 0; i < 2; i++){
        for (j = 0; j < 128; j++){
            linear_para[i][j] = weight3[128*i + j];
            // printf("linear parameter[%d][%d] = %f\n", i, j, weight3[128*i + j]);
        }
    }

    // convolution
    int row, col, filter_num;
    float res;
    for (filter_num = 0; filter_num < 64; filter_num++){
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
    float input2[64][22][22];
    for (k = 0; k < 64; k++){
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
    float output2[64][10][10];
    int stride2 = 2;
    float maxpool;
    for (k = 0; k < 64; k++){
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
    float output4[128];
    for (i = 0; i < 128; i++){
        output4[i] = bias2[i] + MultiplyByElement10(filter2[i], output2, 64);
        if (output4[i] < 0){
            output4[i] = 0.0;
        }
        // printf("output4[%d] = %f\n", i, output4[i]);
    }

    // linear
    float output7[2];
    for (i = 0; i < 2; i++){
        output7[i] = 0;
        for (j = 0; j < 128; j++){
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

    return out[0];
}
