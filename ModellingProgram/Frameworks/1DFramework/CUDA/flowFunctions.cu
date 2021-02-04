#include <cuda.h>
#include <iostream>

/*
__global__
void flowKernel(float *heads, float *queue, int length, float conductivity, float timeDelta, float scale)
{

    int start = blockDim.x*blockIdx.x + threadIdx.x;
    int stop = threadIdx.x + blockDim.x;
    int end = threadIdx.x + blockIdx.x*blockDim.x;
    float mult = conductivity * timeDelta * scale;

    if (start != 0 && stop != length-1)
    {
        // Main stuff code
        for (int i = start; i < end; i++)
        {
            queue[i] =  mult*(heads[i + 1] - heads[i]);    // Right
            queue[i] += mult*(heads[i - 1] - heads[i]);    // Left
        }
    }
    else if (start == 0)
    {
        // First block
        for (int i=1; i < blockDim.x; i++)
        {
            queue[i] =  mult*(heads[i + 1] - heads[i]);    // Right
            queue[i] += mult*(heads[i - 1] - heads[i]);    // Left
        }
        queue[0] = mult*(heads[1] - heads[0]);
    }
    else if (end == length-1)
    {
        for (int i=start; i < range - 1; i++)
        {
          queue[i] =  mult*(heads[i + 1] - heads[i]);    // Right
          queue[i] += mult*(heads[i - 1] - heads[i]);    // Left
        }
        queue[end - 1] = mult*(heads[end - 2] - heads[end-1]);
    }
}
*/

void printArray(float *arr, int length);

__global__
void flowKernel2(float *heads, float *queue, int length, float mult, int elementsPerThread)
{
  int start = elementsPerThread*(blockDim.x*blockIdx.x + threadIdx.x);
  int stop = elementsPerThread*threadIdx.x + elementsPerThread;
  int finalElement = length - 1;

  if (start != 0 && stop != finalElement)
  {
    for (int i = start; i < stop; i++)
    {
      queue[i] =  mult*(heads[i + 1] - heads[i]);
      queue[i] += mult*(heads[i - 1] - heads[i]);
    }
  }
  else if (start == 0 && stop != finalElement)
  {
    for (int i = 1; i < stop; i++)
    {
      queue[i] =  mult*(heads[i + 1] - heads[i]);
      queue[i] += mult*(heads[i - 1] - heads[i]);
    }
    queue[0] = mult*(heads[1] - heads[0]);
  }
  else if (stop == finalElement && start !=0)
  {
    for (int i = start; i < finalElement - 1; i++)
    {
      queue[i] =  mult*(heads[i + 1] - heads[i]);
      queue[i] += mult*(heads[i - 1] - heads[i]);
    }
    queue[finalElement] = mult*(heads[finalElement - 1] - heads[finalElement]);
  }
  else
  {
    for (int i = 1; i < finalElement - 1; i++)
    {
      queue[i] =  mult*(heads[i + 1] - heads[i]);
      queue[i] += mult*(heads[i - 1] - heads[i]);
    }
    queue[finalElement] = mult*(heads[finalElement - 1] - heads[finalElement]);
    queue[0] = mult*(heads[1] - heads[0]);
  }

}

__global__
void diagnosticKernel(float *heads, float *queue, int length, float mult, int elementsPerThread)
{
  int start = elementsPerThread*(blockDim.x*blockIdx.x + threadIdx.x);
  int stop = elementsPerThread*threadIdx.x + elementsPerThread;
  int finalElement = length - 1;

  heads[0] = (float)start;
  heads[1] = (float)stop;
  heads[2] = (float)finalElement;
}
void add(float *heads, float *queue, int length)
{

  printArray(queue,length);
  for (int i = 0; i < length; i++)
  {
    heads[i] += queue[i];
    queue[i] = 0.0;
  }
}

void flow(float *heads, float *queue, int length, float mult, int numBlocks, int threads)
{
  int elementsPerThread = length / (numBlocks*threads);

  float *headsDev;
  float *queueDev;

  std::cout << "Diagnostic data:" << std::endl;
  std::cout << "elementsPerThread: " << elementsPerThread << std::endl;

  const int arrayMemory = length*sizeof(float);

  //std::cout << "Starting malloc" << std::endl;

  cudaMalloc(&headsDev, arrayMemory);
  cudaMalloc(&queueDev, arrayMemory);

  cudaMemcpy(headsDev,heads,arrayMemory,cudaMemcpyHostToDevice);
  cudaMemcpy(queueDev,queue,arrayMemory,cudaMemcpyHostToDevice);

  //std::cout << "Starting kernel" << std::endl;

  flowKernel2<<<numBlocks,threads>>>(headsDev,queueDev,length,mult,elementsPerThread);

  //std::cout << "Finished kernel" << std::endl;

  cudaMemcpy(heads,headsDev,arrayMemory,cudaMemcpyDeviceToHost);
  cudaMemcpy(queue,queueDev,arrayMemory,cudaMemcpyDeviceToHost);

  //cudaFree(heads);
  //cudaFree(queue);

  add(heads,queue,length);

  //std::cout << "finished adding" << std::endl;

  //cudaFree(heads);
  //cudaFree(queue);
}

void printArray(float *arr, int length)
{
  for (int i = 0; i < length; i++)
  {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

int main()
{
  const int length = 4;
  float mult = .1;
  int numBlocks = 1;
  int threads = 1;

  float *heads = new float [length];
  float *queue = new float [length];


  for (int i = 0; i < length; i++)
  {
    heads[i] = 0;
  }

  for (int i = 0; i < length; i++)
  {
    queue[i] = 0;
  }


  heads[0] = 1.0;
  heads[1] = 1.0;
  heads[2] = 1.0;
  heads[3] = 1.0;

  std::cout << "Starting function" << std::endl;
  flow(heads, queue, length, mult, numBlocks, threads);
  std::cout << "Ending kernel" << std::endl;

  printArray(heads, length);
}
