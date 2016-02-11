#include <stdio.h>
#include <stdlib.h>

float* test(float *img, int *data){
    int i;
    for (i = 0; i < 10; i++)    img[i] = data[i] / 2.0;
    printf("&img: %p\n", img);
}

int main(void){
    int data[10] = {1,2,3,4,5,6,7,8,9,10};
    float *img = malloc(10 * sizeof(float));
    test(img, data);

    int i;
    for (i = 0; i < 10; i++){
        printf("%f\n", img[i]);
    }
    printf("&img: %p\n", img);
    printf("img[1]: %f\n", img[1]);
    free(img);
    printf("&img: %p\n", img);
    printf("img[1]: %f\n", img[1]);

    return 0;
}
