import matplotlib.pyplot as plt
import numpy as np
from ModelClass import *
from HandlerClasses import *
from Subclasses import *

mod = FlowModel(name="My first model")

print(mod)

mod.Groundwater.setDefaults()
mod.flow(iters = 1000, track = True, step = 200)

mod.Grapher.plotFlowData()
