#include <iostream>
#include <math.h>
#include <chrono>

using namespace std::chrono;

/*
My GPU has 14 streamining multiprocessors, so a good number of blocks
could be 14, but that's not a good divisible number
*/

/*
So, this function actually is not written to run on the GPU yet.
The function computes ALL of the values instead of just a single one.
*/
__global__
void getQueueGPU(float *heads, float *queue, int length, int stride)
{
  int depth = length/stride;
  int sizeIndex = length-1;


  // Main body
  for (int row = 1; row <= depth - 2; row++)
  {
    for (int i = stride*row + 1; i < stride*(row + 1) - 1; i++)
    {
      queue[i] =  heads[i - stride] - heads[i];    // Up
      queue[i] += heads[i + stride] - heads[i];    // Down
      queue[i] += heads[i - 1] - heads[i];    // Left
      queue[i] += heads[i + 1] - heads[i];    // Right
    }
  }

  //std::cout << "Made it past block 1" << std::endl;

  // Top and bottom edges
  for (int i = 1; i < stride - 1; i++)
  {
    // Top
    queue[i] = heads[i + stride] - heads[i];   // Down
    queue[i] += heads[i - 1] - heads[i];   // left
    queue[i] += heads[i + 1] - heads[i];   // Right

    // Bottom
    queue[sizeIndex - i] = heads[sizeIndex - stride - i] - heads[i];    // Down
    queue[sizeIndex - i] += heads[sizeIndex - i - 1] - heads[i];   // Left
    queue[sizeIndex - i] += heads[sizeIndex - i + 1] - heads[i];   // Right
  }

  //std::cout << "Made it past block 2" << std::endl;

  // Left and right edges
  for (int row = 1; row < stride - 1; row++)
  {
      // Left
      queue[row*stride] = heads[row*stride + 1] - heads[row*stride];    // Right
      queue[row*stride] += heads[(row - 1)*stride] - heads[row*stride];    // Up
      queue[row*stride] += heads[(row + 1)*stride] - heads[row*stride];    // Down

      // Right
      queue[stride*(row + 1) - 1] =  heads[stride*(row + 1) - 2] - heads[stride*(row + 1) - 1];    // Left
      queue[stride*(row + 1) - 1] += heads[stride*(row + 2) - 1] - heads[stride*(row + 1) - 1];    // Up
      queue[stride*(row + 1) - 1] += heads[stride*(row) - 1] - heads[stride*(row + 1) - 1];    // Down
  }

  //std::cout << "Made it past block 3" << std::endl;

  // Corners
  queue[0] =  heads[stride] - heads[0];
  queue[0] += heads[1] - heads[0];

  queue[stride-1] =  heads[2*stride - 1] - heads[stride - 1];
  queue[stride-1] += heads[stride - 2] - heads[stride - 1];

  queue[sizeIndex - stride + 1] =  heads[sizeIndex - 2*stride + 1] - heads[sizeIndex - stride + 1];
  queue[sizeIndex - stride + 1] += heads[sizeIndex - stride + 2] - heads[sizeIndex - stride + 1];

  queue[sizeIndex] =  heads[sizeIndex - stride] - heads[sizeIndex];
  queue[sizeIndex] += heads[sizeIndex - 1] - heads[sizeIndex];
}

/*
The number of blocks should be a multiple of stride, that may
make it a bunch faster
*/
__global__
void getQueueSingle(float *heads, float *queue, int length, int stride)
{
  int xId = threadIdx.x;

  // Check if thread is within first stride
  if (xId < stride)
  {
    // Top
    if (xId > 0 && xId < stride - 1)
    {
      queue[xId] = heads[xId + stride] - heads[xId];   // Down
      queue[xId] += heads[xId - 1] - heads[xId];   // left
      queue[xId] += heads[xId + 1] - heads[xId];   // Right
    }
    // top left corner
    else if (xId == 0)
    {
      queue[0] =  heads[stride] - heads[0];
      queue[0] += heads[1] - heads[0];
    }
    else if (xId == stride - 1)
    {
      queue[stride-1] =  heads[2*stride - 1] - heads[stride - 1];
      queue[stride-1] += heads[stride - 2] - heads[stride - 1];
    }

  }

  // Check if in last stride
  else if (xId > length - stride - 1)
  {
    // Bottom edge
    if (xId > length - stride && xId < length - 1)
    {
      queue[xId] = heads[xId - stride] - heads[xId];    // Down
      queue[xId] += heads[xId - 1] - heads[xId];   // Left
      queue[xId] += heads[xId + 1] - heads[xId];   // Right
    }
    // Bottom left corner
    else if (xId == length - 1 - stride)
    {
      queue[xId] =  heads[xId - stride] - heads[xId];   // Down
      queue[xId] += heads[xId + 1] - heads[xId];        // Right
    }
    // Bottom right
    else if (xId == length - 1)
    {
      queue[xId] =  heads[xId - stride] - heads[xId];   // Down
      queue[xId] += heads[xId - 1] - heads[xId];        // Left
    }

  }

  // Check along left edge
  else if (xId % stride == 0)
  {
    queue[xId] = heads[xId + 1] - heads[xId];          // Right
    queue[xId] += heads[xId + stride] - heads[xId];    // Up
    queue[xId] += heads[xId - stride] - heads[xId];    // Down
  }

  // Check along right edge
  else if ((xId + 1) % stride == 0)
  {
    queue[xId] = heads[xId - 1] - heads[xId];          // Left
    queue[xId] += heads[xId + stride] - heads[xId];    // Up
    queue[xId] += heads[xId - stride] - heads[xId];    // Down
  }

  // Execute main body
  else
  {
    queue[xId] =  heads[xId - stride] - heads[xId];    // Up
    queue[xId] += heads[xId + stride] - heads[xId];    // Down
    queue[xId] += heads[xId - 1] - heads[xId];    // Left
    queue[xId] += heads[xId + 1] - heads[xId];    // Right
  }
}

void printArray(float *arr, int length, int depth)
{
  for (int i = 0; i < depth; i++)
  {
    for (int j = 0; j < length; j++)
    {
      std::cout << arr[i*length + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(void)
{
  const int threadsPerBlock = 256;

  const int length = 1024;
  const int depth = 1024;
  const int arrayLength = length*depth;
  const int numBlocks = arrayLength/threadsPerBlock;

  float *heads;
  float *queue;


  int arrayMemory = sizeof(float)*length*depth;

  cudaMallocManaged(&heads, arrayMemory);
  cudaMallocManaged(&queue, arrayMemory);

  for (int i = 0; i < depth*length; i += (length + 1))
  {
    heads[i] = 1.0f;
  }

  for (int i = 0; i < arrayLength; i++)
  {
    queue[i] = 0.0;
  }

  std::cout << heads[0] << std::endl;

  auto start = high_resolution_clock::now();
  for (int i = 0; i < 1000; i++)
  {
    getQueueSingle<<<numBlocks, threadsPerBlock>>>(heads, queue, arrayLength, length);
    cudaDeviceSynchronize();
  }
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
  float time = (float)duration.count()/(1000000.0f);
  std::cout << "Total execution time for GPU compute: " << time << std::endl;


  cudaFree(heads);
  cudaFree(queue);

}
