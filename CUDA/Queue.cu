#include <iostream>
#include <chrono>

using namespace std::chrono;

__global__ void queueKernel(float *heads, float *queue, int length)
{
  int id = blockDim.x*blockIdx.x + threadIdx.x;
  if (id == 0) heads[id] = 1.0f;
  int final = length - 1;

  // Check to make sure it's not either end
  if ((id != 0) && (id != final))
  {
      queue[id] =  .1*(heads[id + 1] - heads[id]);
      queue[id] += .1*(heads[id - 1] - heads[id]);
  } else if (id == 0)
  {
    queue[0] = 1.0;
    queue[0] = .1*(heads[1] - heads[0]);
  } else if (id == final)
  {
    queue[final] = .1*(heads[final-1]-heads[final]);
  }
}

__global__
void addKernel(float *a, float *b, int length)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  if (id < length) a[id] += b[id];
}

void printArray(float *arr, int length)
{
    for (int i = 0; i < length; i++)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void getQueue(float *heads, float *queue, int length, int iters)
{
  bool debug = false;
  const int num = length*sizeof(float);
  // Create device variables
  float *h, *q;
  // Allocate memory
  cudaMalloc(&h, num);
  cudaMalloc(&q, num);
  // Copy to device
  cudaMemcpy(h, heads, num, cudaMemcpyHostToDevice);
  cudaMemcpy(q, queue, num, cudaMemcpyHostToDevice);
  // Run kernels
  for (int i = 0; i < iters; i++)
  {
    queueKernel<<<length/1024,1024>>> (h, q, length);
    addKernel<<<length/1024,1024>>> (h,q,length);
  }
  // Copy back
  cudaMemcpy(heads, h, num, cudaMemcpyDeviceToHost);
  cudaMemcpy(queue, q, num, cudaMemcpyDeviceToHost);
  cudaFree(h);
  cudaFree(q);

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
    //printArray(queue);
}


int main()
{
    const int length = 1024*1024*64;
    int iters = 5000;
    const int mem = length*sizeof(mem);
    std::cout << iters << " iters" << std::endl;
    std::cout << "------------------------------" << std::endl;

    float *heads, *queue, *heads2, *queue2;
    heads = (float*)malloc(mem);
    queue = (float*)malloc(mem);
    heads2 = (float*)malloc(mem);
    queue2 = (float*)malloc(mem);

    for (int i = 0; i < length; i++)
    {
      heads[i] = 0.0f;
      heads2[i] = 0.0f;
      queue[i] = 0.0f;
      queue2[i] = 0.0f;
    }

    heads[0] = 1.0;

    //printArray(heads,length);
    auto start = high_resolution_clock::now();
    getQueue(heads,queue,length, iters);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    float time = (float)duration.count()/(1000000.0f);
    std::cout << "Total execution time for GPU compute: " << time << std::endl;



    auto start2 = high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
      heads2[0] = 1;
      getQueueCPU(heads2,queue2,length);
    }
    auto stop2 = high_resolution_clock::now();

    auto duration2 = duration_cast<microseconds>(stop2 - start2);
    float time2 = (float)duration2.count()/(1000000.0f);
    std::cout << "Total execution time for CPU compute: " << time2 << std::endl;

    float diff = time2/time;

    std::cout << "GPU function is " << diff << " times faster than CPU function" << std::endl;

    float error = 0;
    for (int i = 0; i < length; i++)
    {
      error += abs(heads[i] - heads2[i]);
    }
    std::cout << "Error: " << error << std::endl;

    free(heads);
    free(queue);
    free(heads2);
    free(queue2);

    std::cout << "\n" << std::endl;

    return 0;
}
