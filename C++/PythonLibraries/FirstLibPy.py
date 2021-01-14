print('Starting')
from ctypes import cdll
try:
    print('Before trying')
    lib = cdll.LoadLibrary('./firstlib.so')
    print('After tried')
except:
    print('Error :(')

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
