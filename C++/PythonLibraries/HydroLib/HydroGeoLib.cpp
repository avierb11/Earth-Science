#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

DLLEXPORT void getQueue1D(float *heads, float *queue, int length, float k, float scale)
{
    /* This function determines the queue for
     * a 1D flow model. Additionally, the conductivity
     * is a constant for the model. Haven't yet implemented
     * a variables conductivity yet, but I will.
     * I've tried to put it in a form that I can translate to CUDA
     */
    int finalIndex = length - 1;
    float multiplier = k/scale;
    for (int i = 0; i < finalIndex; i++)
    {
        if(i!=0 && i!=finalIndex)
        {
            queue[i] =  multiplier*(heads[i - 1] - heads[i]);    // Left
            queue[i] += multiplier*(heads[i + 1] - heads[i]);    // Right
        } else if (i == 0)
        {
            queue[0] = multiplier*(heads[1] - heads[0]);
        } else if (i == finalIndex)
        {
            queue[finalIndex] = multiplier*(heads[finalIndex-1] - heads[finalIndex]);
        }
    }
}
