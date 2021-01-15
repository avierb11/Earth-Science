import ctypes
from ctypes import c_void_p as ptr
import numpy as np

try:
    lib = ctypes.cdll.LoadLibrary('./say.so')
    #length = 700
    #heads = np.identity(length, dtype = np.single)
    #queue = np.zeros((length,length), dtype = np.single)
    #lib.getQueue1Dptr(ptr(heads.ctypes.data),ptr(queue.ctypes.data),length,length)

except:
    print('Module not loaded')

lib = ctypes.cdll.LoadLibrary('./say.so')

#getQueue(heads,queue, length, length)
'''
try:
    lib.getQueue(ptr(heads.ctypes.data),ptr(queue.ctypes.data),length,length)
except:
    print("nope, calculation didn't work")
'''

#print(heads)
#print(np.round(queue, 3))
