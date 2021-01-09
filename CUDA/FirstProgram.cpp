#include <iostream>
#include <math.h>
#include <chrono>

__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
  {
    y[i] = x[i] + y[i];
  }
}

int main(void)
{
  int N = 1<<20;   // specifies 1 million elements

  // Initialize x and y arrays on host (CPU)
  float *x, float *y;

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
  add<<<1, 1>>>(N,x,y);

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
