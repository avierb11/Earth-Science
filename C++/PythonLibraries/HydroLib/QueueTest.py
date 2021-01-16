from ctypes import cdll

try:
    lib = cdll.LoadLibrary('./hydro.so')
    #length = 700
    #heads = np.identity(length, dtype = np.single)
    #queue = np.zeros((length,length), dtype = np.single)
    #lib.getQueue1Dptr(ptr(heads.ctypes.data),ptr(queue.ctypes.data),length,length)
    print("say.so successfully loaded")
except:
    print('Module not loaded')

print("made it to the end")

#getQueue(heads,queue, length, length)
'''
try:
    lib.getQueue(ptr(heads.ctypes.data),ptr(queue.ctypes.data),length,length)
except:
    print("nope, calculation didn't work")
'''

#print(heads)
#print(np.round(queue, 3))
