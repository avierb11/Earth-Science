#include <iostream>
#include <chrono>

using namespace std::chrono;

__global__
void getQueueKernel(float *heads, float *queue, int final)
{
    int id = threadIdx.x;

    // Check to make sure it's not either end
    if (id != 0 and id != final)
    {
        queue[id] =  .1*(heads[id + 1] - heads[id]);
        queue[id] += .1*(heads[id - 1] - heads[id]);
    }
}

void printArray(float *arr, int length)
{
    for (int i = 0; i < length; i++)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void getQueue(float *heads, float *queue, int length)
{
    float *queue = new *float [length];
    int final = length - 1;

    const num = length*sizeof(float);

    cudaMallocManaged(&heads,num);
    cudaMallocManaged(&queue,num);

    getQueueKernel<<<length,1>>>(heads,queue,length);
    cudaFree(heads);
    del queue;
}

void getQueueCPU(float *heads, float *queue, int length)
{
    for (int i = 1; i < length - 1; i++)
    {
        queue[i] = .1*(heads[i+1]-heads[i]);
        queue[i] += .1*(heads[i-1]-heads[i]);
    }
    queue[0] = .1*(heads[1]-heads[0]);
    queue[length-1] = .1*(heads[length-2]-heads[length-1]);

    for (int i = 0; i < length; i++)
    {
        heads[i] += queue[i];
    }
}


int main()
{
    const int length = 5;
    int iters = 1;
    std::cout << iters << " iters" << std::endl;
    std::cout << "------------------------------" << std::endl;

    float *heads = new *float [length];
    float *heads2 = new *float [length];
    float *queue = new *float [length];
    float *queue2 = new *float [length];

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
            getQueue(heads,queue,length);
    }
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    float time = (float)duration.count()/(1000000.0f);
    std::cout << "Total execution time for GPU compute: " << time << std::endl;

    auto start2 = high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        getQueue(heads2,queue2,length);
    }
    auto stop2 = high_resolution_clock::now();

    auto duration2 = duration_cast<microseconds>(stop2 - start2);
    float time2 = (float)duration2.count()/(1000000.0f);
    std::cout << "Total execution time for CPU compute: " << time2 << std::endl;

    return 0;
}
