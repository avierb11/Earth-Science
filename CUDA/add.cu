
#include <iostream>
#include <math.h>
#include <chrono>

using namespace std::chrono;

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y, float *z)
{
int index = threadIdx.x;
int stride = blockDim.x;
for (int i = index; i < n; i += stride)
    z[i] = x[i] + y[i];
}

void addNaive(int n, float *x, float *y, float *z)
{
  for (int i = 0; i < n; i++)
  {
    z[i] = x[i] + y[i];
  }
}

int main(void)
{
//int blocks = 1024*2;
//int threads = 1024;
int N = 1<<23;
float *x, *y, *z;

// Allocate Unified Memory – accessible from CPU or GPU
cudaMallocManaged(&x, N*sizeof(float));
cudaMallocManaged(&y, N*sizeof(float));
cudaMallocManaged(&z, N*sizeof(float));

// initialize x and y arrays on the host
for (int i = 0; i < N; i++) {
  x[i] = 1.0f;
  y[i] = 2.0f;
}

// run first function
auto start0 = high_resolution_clock::now();
for (int i = 0; i < 5000; i++)
{
  addNaive(N, x, y, z);
}
auto stop0 = high_resolution_clock::now();



auto duration0 = duration_cast<microseconds>(stop0 - start0);
std::cout << duration0.count() << " for CPU function" << std::endl;

// Reset values
for (int i = 0; i < N; i++) {
  x[i] = 1.0f;
  y[i] = 2.0f;
}
auto start = high_resolution_clock::now();
// Run kernel on 1M elements on the GPU
int count = 0;
for (int i = 0; i < 5000; i++)
{
  add<<<8, 512>>>(N, x, y, z);
  count++;
}
// Wait for GPU to finish before accessing on host
cudaDeviceSynchronize();
auto stop = high_resolution_clock::now();

std::cout << "count: " << count << std::endl;


// Check for errors (all values should be 3.0f)

float maxError = 0.0f;
for (int i = 0; i < N; i++)
  maxError = fmax(maxError, fabs(z[i]-3.0f));
std::cout << "Max error: " << maxError << std::endl;

auto duration = duration_cast<microseconds>(stop - start);
float time = (float)duration.count()/1000000;
std::cout << duration.count()<< " for GPU function" << std::endl;

//std::cout << time << " seconds" << std::endl;
// Free memory
cudaFree(x);
cudaFree(y);

float multiplier;
multiplier = (float)duration0.count()/(float)duration.count();

std::cout << "GPU computing is " << multiplier << " times faster" << std::endl;

return 0;
}
