#include <iostream>
#include <vector>
#include <chrono>

using std::vector;
using namespace std::chrono;

/*
The function for determining queue. I have done this function in python and C#,
but not yet C++
*/

void getQueue(vector< vector<float> > &heads, vector< vector< float> > &queue);
void printVector(vector< vector<float> > &vec);
void printVector1D(vector<float> &vec, int width, int depth);
void getQueue1D(vector<float> &heads, vector<float> &queue, int stride);
void getDischarge2D(float *heads, float *queue, float *conductivity, float *heights, int stride, float xScale = 1.0f, float yScale = 1.0f);
void printArray(float *x, int length, int stride);
void getQueue1Dptr(float *heads, float *queue, int length, int stride);


int main()
{
  const int depth = 8192*2;
  const int length = 8192*2;
  const int totalLength = depth*length;

  // 1D vector
  /*
  vector<float> heads2 (length*depth, 0);
  for (int i = 0; i < depth*length; i += (length + 1))
  {
    heads2[i] = 1.0f;
  }
  vector<float> queue2 (length*depth, 0);
  */
  float *heads = new float [length*depth];
  float *queue = new float[length*depth];

  for (int i = 0; i < length*depth; i++)
  {
      heads[i] = 0;
      queue[i] = 0;
  }

  for (int i = 0; i < length; i++)
  {
      heads[i*depth + i] = 1.0f;
  }

  // 1D array timing
  auto start = high_resolution_clock::now();
  for (int i = 0; i < 1; i++)
  {
    getQueue1Dptr(heads, queue, totalLength, length);
  }
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
  float time = (float)duration.count()/1000000.0f;
  std::cout << "Total execution time for 1D array: " << time << std::endl;

  //delete []point;

  std::cout << "made it to the end of the program" << std::endl;

  return 0;
}

void getQueue(vector< vector<float> > &heads, vector< vector< float> > &queue)
{
  int depth = heads.size();
  int length = heads[0].size();
  int finalDepth = depth - 1;
  int finalLength = length - 1;

  // Begin with interior region
  for (int i = 1; i < depth-1; i++)
  {
    for (int j = 1; j < length-1; j++)
    {
      queue[i][j] =  heads[i - 1][j] - heads[i][j];    // Up
      queue[i][j] += heads[i + 1][j] - heads[i][j];    // Down
      queue[i][j] += heads[i][j - 1] - heads[i][j];    // Left
      queue[i][j] += heads[i][j + 1] - heads[i][j];    // Right
    }
  }

  // Top and bottom edges
  for (int j = 1; j < length - 1; j++)
  {
    queue[0][j] =  heads[0][j - 1] - heads[0][j];    // Left
    queue[0][j] += heads[0][j + 1] - heads[0][j];    // Right
    queue[0][j] += heads[1][j] - heads[0][j];        // Down

    queue[finalDepth][j] =  heads[finalDepth][j - 1] - heads[finalDepth][j];   // Left
    queue[finalDepth][j] += heads[finalDepth][j + 1] - heads[finalDepth][j];   // Right
    queue[finalDepth][j] += heads[finalDepth - 1][j] - heads[finalDepth][j];   // Up
  }

  // Left and right edges
  for (int i = 1; i < depth - 1; i++)
  {
    queue[i][0] =  heads[i + 1][0] - heads[i][0];    // Up
    queue[i][0] += heads[i - 1][0] - heads[i][0];    // Down
    queue[i][0] += heads[i][1] - heads[i][0];        // Right

    queue[i][finalLength] =  heads[i - 1][finalLength] - heads[i][finalLength];  // Up
    queue[i][finalLength] += heads[i + 1][finalLength] - heads[i][finalLength];  // Down
    queue[i][finalLength] += heads[i][finalLength - 1] - heads[i][finalLength];  // left
  }

  // Corners
  // Top left
  queue[0][0] =  heads[1][0] - heads[0][0];
  queue[0][0] += heads[0][1] - heads[0][0];

  // Top Right
  queue[0][finalLength] =  heads[1][finalLength] - heads[0][finalLength];
  queue[0][finalLength] += heads[0][finalLength - 1] - heads[0][finalLength];

  // Bottom Left
  queue[finalDepth][0] =  heads[finalDepth - 1][0] - heads[finalDepth][0];
  queue[finalDepth][0] += heads[finalDepth][1] - heads[finalDepth][0];

  // Bottom Right
  queue[finalDepth][finalLength] =  heads[finalDepth - 1][finalLength] - heads[finalDepth][finalLength];
  queue[finalDepth][finalLength] += heads[finalDepth][finalLength - 1] - heads[finalDepth][finalLength];

}

void getQueue1D(vector<float> &heads, vector<float> &queue, int stride)
{
  int depth = heads.size()/stride;
  int sizeIndex = heads.size() - 1;


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

void getQueue1Dptr(float *heads, float *queue, int length, int stride)
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

void printVector(vector< vector<float> > &vec)
{
  for (int j = 0; j < vec.size(); j++)
  {
    for (int i = 0; i < vec[0].size(); i++)
    {
      std::cout << vec[j][i] << " ";
    }
    std::cout << std::endl;
  }
}

void printVector1D(vector<float> &vec, int width, int depth)
{
  for (int i = 0; i < depth; i++)
  {
    for (int j = 0; j < width; j++)
    {
      std::cout << vec[i*depth + j] << " ";
    }
    std::cout << std::endl;
  }
}

void getDischarge2D(float *heads, float *queue, float *conductivity, float *heights, int length, int stride, float xScale = 1.0f, float yScale = 1.0f)
{
    int depth = length/stride;
    int sizeIndex = length - 1;


    // Main body
    for (int row = 1; row <= depth - 2; row++)
    {
      for (int i = stride*row + 1; i < stride*(row + 1) - 1; i++)
      {
        queue[i] =  (heads[i - stride] - heads[i])*heights[i]/yScale;    // Up
        queue[i] += heads[i + stride] - heads[i]*heights[i]/yScale;    // Down
        queue[i] += heads[i - 1] - heads[i]*heights[i]/xScale;    // Left
        queue[i] += heads[i + 1] - heads[i]*heights[i]/xScale;    // Right
      }
    }

    //std::cout << "Made it past block 1" << std::endl;

    // Top and bottom edges
    for (int i = 1; i < stride - 1; i++)
    {
      // Top
      queue[i] = (heads[i + stride] - heads[i])*heights[i]/yScale;   // Down
      queue[i] += (heads[i - 1] - heads[i])*heights[i]/xScale;   // left
      queue[i] += (heads[i + 1] - heads[i])*heights[i]/xScale;   // Right

      // Bottom
      queue[sizeIndex - i] =  (heads[sizeIndex - stride - i] - heads[i])*heights[i]/yScale;    // Down
      queue[sizeIndex - i] += (heads[sizeIndex - i - 1] - heads[i])*heights[i]/xScale;   // Left
      queue[sizeIndex - i] += (heads[sizeIndex - i + 1] - heads[i])*heights[i]/xScale;   // Right
    }

    //std::cout << "Made it past block 2" << std::endl;

    // Left and right edges
    for (int row = 1; row < stride - 1; row++)
    {
        // Left
        queue[row*stride] = (heads[row*stride + 1] - heads[row*stride])*heights[row*stride]/xScale;    // Right
        queue[row*stride] += (heads[(row - 1)*stride] - heads[row*stride])*heights[row*stride]/yScale;    // Up
        queue[row*stride] += (heads[(row + 1)*stride] - heads[row*stride])*heights[row*stride]/yScale;    // Down

        // Right
        queue[stride*(row + 1) - 1] =  (heads[stride*(row + 1) - 2] - heads[stride*(row + 1) - 1])*heights[row*stride]/xScale;    // Left
        queue[stride*(row + 1) - 1] += (heads[stride*(row + 2) - 1] - heads[stride*(row + 1) - 1])*heights[row*stride]/yScale;    // Up
        queue[stride*(row + 1) - 1] += (heads[stride*(row) - 1] - heads[stride*(row + 1) - 1])*heights[row*stride]/yScale;    // Down
    }

    //std::cout << "Made it past block 3" << std::endl;

    // Corners
    queue[0] =  (heads[stride] - heads[0])*heights[0]/yScale;   // Down
    queue[0] += (heads[1] - heads[0])*heights[0]/xScale;        // Right

    queue[stride-1] =  (heads[2*stride - 1] - heads[stride - 1])*heights[stride-1]/yScale; // Down
    queue[stride-1] += (heads[stride - 2]   - heads[stride - 1])*heights[stride-1]/xScale;       // Left

    queue[sizeIndex - stride + 1] =  (heads[sizeIndex - 2*stride + 1] - heads[sizeIndex - stride + 1])*heights[sizeIndex - stride + 1]/yScale;   // Up
    queue[sizeIndex - stride + 1] += (heads[sizeIndex - stride + 2] - heads[sizeIndex - stride + 1])*heights[sizeIndex - stride + 1]/xScale;     // Right

    queue[sizeIndex] =  (heads[sizeIndex - stride] - heads[sizeIndex])*heights[sizeIndex]/yScale;   // Up
    queue[sizeIndex] += (heads[sizeIndex - 1] - heads[sizeIndex])*heights[sizeIndex]/xScale;        // Left


    // Now, we have the queue, so add the queue to the heads to update it
}

void updateHeads2D(float *heads, float *queue, float *conductivity, float *heights, float *Ss,  int length, int stride, float timeStep=1.0f, float xScale = 1.0f, float yScale = 1.0f)
{
    int depth = length/stride;
    int sizeIndex = length - 1;


    // Main body
    for (int row = 1; row <= depth - 2; row++)
    {
      for (int i = stride*row + 1; i < stride*(row + 1) - 1; i++)
      {
        queue[i] =  (heads[i - stride] - heads[i])*timeStep*heights[i]/yScale;    // Up
        queue[i] += (heads[i + stride] - heads[i])*timeStep*heights[i]/yScale;    // Down
        queue[i] += (heads[i - 1] - heads[i])*timeStep*heights[i]/xScale;    // Left
        queue[i] += (heads[i + 1] - heads[i])*timeStep*heights[i]/xScale;    // Right
      }
    }

    //std::cout << "Made it past block 1" << std::endl;

    // Top and bottom edges
    for (int i = 1; i < stride - 1; i++)
    {
      // Top
      queue[i] = (heads[i + stride] - heads[i])*timeStep*heights[i]/yScale;   // Down
      queue[i] += (heads[i - 1] - heads[i])*timeStep*heights[i]/xScale;   // left
      queue[i] += (heads[i + 1] - heads[i])*timeStep*heights[i]/xScale;   // Right

      // Bottom
      queue[sizeIndex - i] =  (heads[sizeIndex - stride - i] - heads[i])*timeStep*heights[i]/yScale;    // Down
      queue[sizeIndex - i] += (heads[sizeIndex - i - 1] - heads[i])*timeStep*heights[i]/xScale;   // Left
      queue[sizeIndex - i] += (heads[sizeIndex - i + 1] - heads[i])*timeStep*heights[i]/xScale;   // Right
    }

    //std::cout << "Made it past block 2" << std::endl;

    // Left and right edges
    for (int row = 1; row < stride - 1; row++)
    {
        // Left
        queue[row*stride] = (heads[row*stride + 1] - heads[row*stride])*timeStep*heights[row*stride]/xScale;    // Right
        queue[row*stride] += (heads[(row - 1)*stride] - heads[row*stride])*timeStep*heights[row*stride]/yScale;    // Up
        queue[row*stride] += (heads[(row + 1)*stride] - heads[row*stride])*timeStep*heights[row*stride]/yScale;    // Down

        // Right
        queue[stride*(row + 1) - 1] =  (heads[stride*(row + 1) - 2] - heads[stride*(row + 1) - 1])*timeStep*heights[row*stride]/xScale;    // Left
        queue[stride*(row + 1) - 1] += (heads[stride*(row + 2) - 1] - heads[stride*(row + 1) - 1])*timeStep*heights[row*stride]/yScale;    // Up
        queue[stride*(row + 1) - 1] += (heads[stride*(row) - 1] - heads[stride*(row + 1) - 1])*timeStep*heights[row*stride]/yScale;    // Down
    }

    //std::cout << "Made it past block 3" << std::endl;

    // Corners
    queue[0] =  (heads[stride] - heads[0])*timeStep*heights[0]/yScale;   // Down
    queue[0] += (heads[1] - heads[0])*timeStep*heights[0]/xScale;        // Right

    queue[stride-1] =  (heads[2*stride - 1] - heads[stride - 1])*timeStep*heights[stride-1]/yScale; // Down
    queue[stride-1] += (heads[stride - 2]   - heads[stride - 1])*timeStep*heights[stride-1]/xScale;       // Left

    queue[sizeIndex - stride + 1] =  (heads[sizeIndex - 2*stride + 1] - heads[sizeIndex - stride + 1])*timeStep*heights[sizeIndex - stride + 1]/yScale;   // Up
    queue[sizeIndex - stride + 1] += (heads[sizeIndex - stride + 2] - heads[sizeIndex - stride + 1])*timeStep*heights[sizeIndex - stride + 1]/xScale;     // Right

    queue[sizeIndex] =  (heads[sizeIndex - stride] - heads[sizeIndex])*timeStep*heights[sizeIndex]/yScale;   // Up
    queue[sizeIndex] += (heads[sizeIndex - 1] - heads[sizeIndex])*timeStep*heights[sizeIndex]/xScale;        // Left


    // Now, we have the total volume of water, so convert to head change
    for (int i = 0; i < length; i++)
    {
        queue[i] *= 1/(Ss[i] * heights[i] * xScale * yScale);
        heads[i] += queue[i];
    }
}

void printArray(float *x, int length, int stride)
{
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < stride; j++)
        {
            std::cout << x[stride*i + j] << " ";
        }
        std::cout << std::endl;
    }
}
