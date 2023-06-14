#include <stdio.h>

double integrate(double (*f)(double), double a, double b) 
{
    double result = 0.0;
    double dx = 0.0001;
    double x;
    for (x = a; x < b; x += dx) 
    {
        result += f(x) * dx;
    }
    return result;
}

double square(double x) 
{
    return x * x;
}

int main()
{
    printf("integrate(square, 0.0, 2.0) = %f\n",integrate(square, 0.0, 2.0));
}
