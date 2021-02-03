#include <cuda.h>

__global__
void flowKernel(float *heads, float *queue, int length, float conductivity, float timeDelta, float scale)
{
    int range = blockDim.x;

    int start = threadIdx.x;
    int end = threadIdx.x + blockIdx.x*blockDim.x;
    float mult = conductivity * timeDelta * scale;
    float val = 0;

    if (start != 0 && stop != length-1)
    {
        // Main stuff code
        for (int i = start; i < end; i++)
        {
            val = mult*(heads[i + 1] - heads[i]);
            queue[i]      =  val;    // Right
            queue[i + 1]  -= val;    // Left
        }
    }
    else if (start == 0)
    {
        // First block
        for (int i=1; i < blockDim.x; i++)
        {
            val = mult*(heads[i + 1] - heads[i]);
            queue[i]      =  val;    // Right
            queue[i + 1]  -= val;    // Left
        }
        queue[0] = mult*(heads[1] - heads[0])
    }
    else if (end == length-1)
    {
        for (int i=start; i < range - 1; i++)
        {
            val = mult*(heads[i + 1] - heads[i]);
            queue[i]      =  val;    // Right
            queue[i + 1]  -= val;    // Left
        }
        queue[end - 1] = mult*(heads[end - 2] - heads[end-1]);
    }
}
