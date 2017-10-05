#include <stdio.h>
#include <iostream>
#include <math.h>
#include <ctime>

float HALF_RAND_MAX = (float)RAND_MAX / 2.0;
float trand() {
    return (float)std::rand() / HALF_RAND_MAX - 1.0;
}