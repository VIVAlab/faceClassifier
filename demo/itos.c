// convert numbers from int to char

char* itos(int i, char b[]){
    char const digit[] = "0123456789";
    char* p = b;
    int shifter = i;

    //Move to where representation ends
    do{
        ++p;
        shifter = shifter/10;
    }while(shifter);

    *p = '\0';

    //Move back and insert digits
    do{
        *--p = digit[i%10];
        i = i/10;
    }while(i);

    return b;
}
