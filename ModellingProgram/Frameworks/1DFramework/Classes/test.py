import matplotlib.pyplot as plt
import numpy as np
from ModelClass import *
from HandlerClasses import *
from Subclasses import *

mod = FlowModel(name="My first model", numElements = 100)

#mod.Groundwater.setDefaults()
mod.Groundwater.addPump(1, .05)
mod.Groundwater.addWell(8.5, .05)
mod.flow(iters = 150, track = True, step = 30)

mod.Grapher.plotHeads()
