from ctypes import cdll
try:
    lib = cdll.LoadLibrary('./firstlib.so')
    print('successfully loaded library')
except:
    print('Error :(')

try:
    lib2 = cdll.LoadLibrary('./test.so')
    print("successfully loaded test.so")
except:
    print("didn't load test2 :(")

print("Made it!")

'''
class Foo(object):
    def __init__(self):
        self.obj = lib.Foo_new()

    def bar(self):
        lib.Foo_bar(self.obj)


f = Foo()
f.bar()
'''
