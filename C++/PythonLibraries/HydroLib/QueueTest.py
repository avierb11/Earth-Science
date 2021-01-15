import ctypes
from ctypes import c_void_p as ptr
import numpy as np

try:
    lib = ctypes.cdll.LoadLibrary('getQueue.so')
except:
    print('Module not loaded')


length = 700
heads = np.identity(length, dtype = np.single)
queue = np.zeros((length,length), dtype = np.single)

#getQueue(heads,queue, length, length)
'''
try:
    lib.getQueue(ptr(heads.ctypes.data),ptr(queue.ctypes.data),length,length)
except:
    print("nope, calculation didn't work")
'''
lib.getQueue1Dptr(ptr(heads.ctypes.data),ptr(queue.ctypes.data),length,length)

#print(heads)
#print(np.round(queue, 3))
