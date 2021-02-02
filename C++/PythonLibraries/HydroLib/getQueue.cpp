#include <iostream>
#include <chrono>
using namespace std::chrono;

extern "C" void getQueue1Dptr(float *heads, float *queue, int length, int stride)
{
  int depth = length/stride;
  int sizeIndex = length - 1;


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

void printArray(float *x, int length, int stride)
{
  int depth = length/stride;
  for (int i = 0; i < depth; i++)
  {
    for (int j = 0; j < stride; j++)
    {
      std::cout << x[stride*i + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main()
{
  int length = 16384 * 2;
  int arrLength = length*length;
  float *x = new float [length*length];
  for (int i = 0; i < length*length; i++)
  {
    x[i] = 0.0;
  }
  for (int i = 0; i < length; i++)
  {
    x[i*length + i] = 1.0;
  }

  float *y = new float [length*length];
  for (int i = 0; i < length*length; i++)
  {
    y[i] = 0.0;
  }

  auto start = high_resolution_clock::now();
  for (int i = 0; i < 3; i++)
  {
    getQueue1Dptr(x,y,arrLength,length);
  }
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
  float time = (float)duration.count()/1000000.0f;
  std::cout << "Total execution time C++ only: " << time << std::endl;



  std::cout << "made it to the end" << std::endl;


  return 0;
}
