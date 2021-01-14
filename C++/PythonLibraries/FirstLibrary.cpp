
extern "C" int add(int a, int b)
{
  return a + b;
}

int main()
{
  std::cout << "Hello first library" << std::endl;

  int a = 1;
  int b = 2;
  int c = add(a,b);

  std::cout<< "c: " << c << std::endl;

  return 0;
}
