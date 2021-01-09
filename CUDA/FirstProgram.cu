#include <iostream>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = index; i < n; i+= stride)
  {
    y[i] = x[i] + y[i];
  }
}

int main(void)
{
  int N = 1<<22;   // specifies 1 million elements

  // Initialize x and y arrays on host (CPU)
  float *x, *y;

  // Allocate Unified Memory - replaces the new keywork
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // set values in the arrays
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  auto start = high_resolution_clock::now();
  // Run kernel on 1M arrays on CPU
  add<<<128, 512>>>(N,x,y);
  // eventually, use add<<<, 256>>>

  auto stop = high_resolution_clock::now();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(x);
  cudaFree(y);

  auto duration = duration_cast<microseconds>(stop - start);
  float time = (float)duration.count()/1000000;
  cout << duration.count() << endl;

  cout << time << " seconds" << endl;

  return 0;
}
