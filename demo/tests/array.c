#include <stdio.h>
#include <stdlib.h>

void test(float **img, int data[][3], int m, int n){
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            img[i][j] = data[i][j] / 2.0;
}

void freeArray(float **img, int m) {
    int i;
    for (i = 0; i < m; ++i) {
        free(img[i]);
    }
    free(img);
}

int main(void){
    int i, j, m = 2, n = 3;
    float **img = malloc(m * sizeof(float*));

    for (i = 0; i < m; i++){
        img[i] = malloc(n * sizeof(float));
    }

    int data[2][3] = {{1,2,3},{4,5,6}};

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            printf("data: %d\n", data[i][j]);
    test(img, data, m, n);

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            printf("img: %f\n", img[i][j]);

    printf("img[1][2]: %f\n", img[1][2]);
    freeArray(img, 2);
    printf("img[1][2]: %f\n", img[1][2]);

    return 0;
}
