
extern "C" int addSomething(int a, int b)
{
  return a + b;
}

int main()
{
  int a = 1;
  int b = 2;
  int c = addSomething(a,b);
  return 0;
}
