#!/usr/bin/env python
# coding: utf-8

# # Import some stuff

# In[1]:


from ctypes import cdll
from ctypes import c_void_p as ptr
import numpy as np
from timeit import default_timer as timer
lib = cdll.LoadLibrary('./hydroLib.so')


# # Make some variables

# In[9]:


length = 2048
total = length**2
heads = np.identity(length)
queue = np.zeros((length,length))


# # Run the code and time it

# In[10]:


#get_ipython().run_cell_magic('time', '', 'for i in range(1):\n    lib.getQueue1Dptr(ptr(heads.ctypes.data),ptr(queue.ctypes.data),total,length)\n    \nprint("made it to the end")')


# In[11]:
start = timer()
for i in range(100):
    lib.getQueue1Dptr(ptr(heads.ctypes.data),ptr(queue.ctypes.data),total,length)
end = timer()

time = (end-start)
print("Total time:", (end-start))
print("time per iteration:",time/100)


# In[ ]:




