import matplotlib.pyplot as plt
import numpy as np
from ModelClass import *
from HandlerClasses import *
from Subclasses import *

mod = FlowModel(name="My first model", numElements = 100, conductivity = .01, timeDelta = .1)

#mod.Groundwater.setDefaults()
mod.Groundwater.addPump(1, .05)
mod.Groundwater.addWell(8, .05)
mod.flow(iters = 10000, track = True, step = 30)

print(mod.heads)

mod.Grapher.plotHeads()
