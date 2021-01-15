import ctypes
import numpy as np

try:
    lib = ctypes.cdll.LoadLibrary('addlib.so')
except:
    print(':< (')

a = np.ones((2,2), dtype = np.int32)
b = np.full((2,2), 2, dtype = np.int32)
c = np.zeros((2,2), dtype = np.int32)

print("Ctypes data")
try:
    lib.addVec(ctypes.c_void_p(a.ctypes.data), ctypes.c_void_p(b.ctypes.data), ctypes.c_void_p(c.ctypes.data),4)
except:
    print("nope, didn't work")

print(c)

#lib.addVec()

print("Made it to the end")
