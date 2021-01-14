#include <iostream>
#include <windows.h>

int WINAPI DllEntryPoint(HINSTANCE hinst, unsigned long reason, void* lpReserved)
{
  return 1;
}
//---------------------------------------------------------------------------
int WINAPI WinMain(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nCmdShow)
{
  return 0;
}

class Foo{
public:
    void bar(){
        std::cout << "Hello" << std::endl;
    }
};

extern "C" {
    Foo* Foo_new(){ return new Foo();}
    void Foo_bar(Foo* foo){ foo->bar();}
}
