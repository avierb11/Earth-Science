#include <iostream>

extern "C" void addVec(int *a, int *b, int *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

extern "C" void saySomething()
{
    std::cout << "hey, this function worked" << std::endl;
}

extern "C" void printArray(int *array, int length)
{
    for (int i = 0; i < length; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main()
{
    int length = 3;
    int *a = new int [length];
    int *b = new int [length];
    int *c = new int [length];


    std::cout << "Made it to point 1" << std::endl;

    for (int i = 0; i < length; i++)
    {
        a[i] = 1;
        b[i] = 2;
    }

    addVec(a,b,c,length);
    std::cout << "After: " << std::endl;
    printArray(c,length);


    std::cout << "ended" << std::endl;
    return 0;
}
