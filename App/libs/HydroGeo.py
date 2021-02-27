import numpy as np

class Groundwater:
    """
    This class is a container for the array that handles hydraulic head flow
    """
    def __init__(self, xDim = 1000, yDim = 1000, zDim = 100, numElementsX = 100, numElementsY = 100, numElementsZ = 100):
        self.xDim = xDim
        self.yDim = yDim
        self.zDim = zDim
        self.numElementsX = numElementsX
        self.numElementsY = numElementsY
        self.numElementsZ = numElementsZ
        self.xScale = xDim/numElementsX
        self.xScale = yDim/numElementsY
        self.xScale = zDim/numElementsZ
        self.groundwater_array = np.zeros((numElementsX,numElementsY,numElementsZ),dtype = np.single) # The groundwater array

    def GetTransientFlow(self, conductivity, timeDelta):
        do_nothing("Groundwater.GetTransientFlow")

class WellManager:
    """
    This class stores and manages well-related things for the model, whether it be
    creating, storing, and collecting data on wells


    """
    def __init__(self):
        self.Wells = []

class Weather:
    """Determines a weather pattern including:
        - Precipitation (rain, snow, hail, sleet)
        - Temperature
        - Wind
        - Freeze/thaw
    """
    def __init__(self):
        pass

    def Rain(self):
        do_nothing("Weather.Rain")

    def Snow(self):
        do_nothing("Weather.Snow")

def do_nothing(func = "unnamed"):
    '''A placeholder function for when I haven't yet put in a definition'''
    print("{} not yet implemented".format(func))
