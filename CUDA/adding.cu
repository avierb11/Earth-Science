#include <iostream>

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

__global__ void testKernel(float *heads, float *queue, int length)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id < length) queue[id] = .3f;
}

__global__ void queueKernel(float *heads, float *queue, int length)
{
  int id = blockDim.x*blockIdx.x + threadIdx.x;
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

void getQueue(float *heads, float *queue, int length, int iters)
{
  int final = length - 1;
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
    queueKernel<<<length,1>>> (h, q, length);
    addKernel<<<length,1>>> (h,q,length);
  }
  // Copy back
  cudaMemcpy(heads, h, num, cudaMemcpyDeviceToHost);
  cudaMemcpy(queue, q, num, cudaMemcpyDeviceToHost);
  cudaFree(h);
  cudaFree(q);

}

void add(float *a, float *b, int length)
{
  const int mem = length*sizeof(float);
  float *d_a, *d_b;

  // Allocate memory
  cudaMalloc(&d_a,mem);
  cudaMalloc(&d_b,mem);

  cudaMemcpy(d_a,a,mem,cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b,mem,cudaMemcpyHostToDevice);

  addKernel<<<1,length>>>(d_a,d_b,length);

  cudaMemcpy(b,d_b,mem,cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);

}

int main(void)
{
  const int length = 5;
  const int mem = length*sizeof(mem);

  float *heads, *queue;

  heads = (float*)malloc(mem);
  queue = (float*)malloc(mem);

  for (int i = 0; i < length; i++)
  {
    heads[i] = 0.0f;
    queue[i] = 0.0f;
  }
  heads[0] = 1.0f;

  getQueue(heads,queue,length, 1);
  std::cout<<"heads: ";
  printArray(heads,length);


  free(heads);
  free(queue);

  return 0;
}
