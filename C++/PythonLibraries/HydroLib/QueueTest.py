from ctypes import cdll
from ctypes import c_void_p as ptr
import numpy as np

try:
    lib = cdll.LoadLibrary('./hydroLib.so')
    length = 1024*2
    heads = np.identity(length, dtype = np.single)
    queue = np.zeros((length,length), dtype = np.single)
    lib.getQueue1Dptr(ptr(heads.ctypes.data),ptr(queue.ctypes.data),length*length,length)
    print("Module successfully loaded\n")
except:
    print('Module not loaded')

print('Function all worked')
