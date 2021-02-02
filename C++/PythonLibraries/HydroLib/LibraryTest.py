import ctypes
from ctypes import *
import numpy as np
from time import perf_counter

def get_getQueue():
    lib = ctypes.windll.LoadLibrary("./dlls/HydroGeoLib.dll")
    func = lib.getQueue1D
    func.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_float, c_float]
    return func

__getQueue1D = get_getQueue()

#-----------------------------------------------

def getQueue(a,b,k,scale):
    A = a.ctypes.data_as(POINTER(c_float))
    B = b.ctypes.data_as(POINTER(c_float))
    k = float(k)
    scale = float(scale)
    length = int(a.size)

    __getQueue1D(A,B,length,k,scale)


def getQueueN(heads,queue,k,scale):
    multiplier = k/scale
    queue[:-1] =  heads[1: ] - heads[:-1]
    queue[1: ] += heads[:-1] - heads[1: ]
    queue *= multiplier

#----------------------------------------------
# Numpy first
heads = np.linspace(1,0,1024).astype('float32')
queue = np.zeros(1024).astype('float32')
k = .01
scale = .5
iters = 10000

time1Start = perf_counter()
for i in range(iters):
    getQueueN(heads,queue,k,scale)
time1Stop = perf_counter()
time1 = time1Stop-time1Start

#-----------------------------------------------
# C function after


heads = np.linspace(1,0,1024).astype('float32')
queue = np.zeros(1024).astype('float32')

time2Start = perf_counter()
for i in range(iters):
    getQueue(heads,queue,k,scale)
time2Stop = perf_counter()
time2 = time2Stop-time2Start

#-----------------------------------------------
print('Execution times')
print('-----------------------------------------')
print('Time for numpy only function:',time1)
print('Time for C++ function:',time2)
print('\n')
