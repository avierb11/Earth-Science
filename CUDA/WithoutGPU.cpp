#include <iostream>
#include <chrono>
#include <Vector>
#include <math.h>

using namespace std;
using namespace std::chrono;

void add(vector<float> &a, vector<float> &b, vector<float> &c);

int main()
{
  int length = pow(2,22);

  vector<float> A(length);
  vector<float> B(length);
  vector<float> C(length);
  for (int i = 0; i < length; i++)
  {
    A[i] = 1.0;
    B[i] = 2.0;
  }

  auto start = high_resolution_clock::now();

  // Execute function
  add(A, B, C);

  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
  float time = (float)duration.count()/1000000;
  cout << duration.count() << endl;

  cout << time << " seconds" << endl;

}

void add(vector<float> &a, vector<float> &b, vector<float> &c)
{
  for (int i = 0; i < a.size(); i++)
  {
    c[i] = a[i]  + b[i];
  }
}
