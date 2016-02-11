// multi-dimensional array mulitplication
// function reload??

float MultiplyByElement3(float m1[][3], float m2[][3]){
    int i,j;
    float result = 0.0;

    for (i = 0; i < 3; i++){
        for (j = 0; j < 3; j++){
            result = m1[i][j] * m2[i][j] + result;
        }
    }

    return result;
}

float MultiplyByElement5(float m1[][5][5], float m2[][5][5], int outer_loop){
    int i,j,k;
    float result = 0.0;

    for (k = 0; k < outer_loop; k++){
        for (i = 0; i < 5; i++){
            for (j = 0; j < 5; j++){
                result = m1[k][i][j] * m2[k][i][j] + result;
            }
        }
    }

    return result;
}

// MultiplyByElement5 without outer_loop
float MultiplyByElement5_(float m1[][5], float m2[][5]){
    int i,j;
    float result = 0.0;

    for (i = 0; i < 5; i++){
        for (j = 0; j < 5; j++){
            result = m1[i][j] * m2[i][j] + result;
        }
    }

    return result;
}

float MultiplyByElement9(float m1[][9][9], float m2[][9][9], int outer_loop){
    int i,j,k;
    float result = 0.0;

    for (k = 0; k < outer_loop; k++){
        for (i = 0; i < 9; i++){
            for (j = 0; j < 9; j++){
                result = m1[k][i][j] * m2[k][i][j] + result;
            }
        }
    }

    return result;
}

float MultiplyByElement10(float m1[][10][10], float m2[][10][10], int outer_loop){
    int i,j,k;
    float result = 0.0;

    for (k = 0; k < outer_loop; k++){
        for (i = 0; i < 10; i++){
            for (j = 0; j < 10; j++){
                result = m1[k][i][j] * m2[k][i][j] + result;
            }
        }
    }

    return result;
}

float MultiplyByElement18(float *m1, float m2[][18][18], int outer_loop){
    int i,j,k;
    float result = 0.0;

    for (k = 0; k < outer_loop; k++){
        for (i = 0; i < 18; i++){
            for (j = 0; j < 18; j++){
                result = m1[18*18*k+18*i+j] * m2[k][i][j] + result;
            }
        }
    }

    return result;
}
