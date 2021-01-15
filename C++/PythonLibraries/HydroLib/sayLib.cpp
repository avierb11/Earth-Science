#include <iostream>

extern "C" void saySomething()
{
  std::cout << "Made it here" << std::endl;
}

int main()
{
  return 0;
}
