'''
This file defines the full modelling class that
organizes the various other classes
'''
import numpy as np
import matplotlib.pyplot as plt
from HandlerClasses import *
from Subclasses import *


class FlowModel:
    def __init__(self, name = "Unnamed flow model", length = 10, numElements = 10, timeDelta = 1, conductivity = .1, specificStorage = .25, porosity = .25):
        # Basic elements
        self.length = length
        self.name = name
        self.numElements = numElements
        self.timeDelta = timeDelta
        self.scale = length/numElements
        self.time = 0   # Measures the total time from the inception of the model
        self.lengthPoints = np.linspace(0,length,numElements)

        # Groundwater things
        self.conductivity = conductivity
        self.specificStorage = specificStorage
        self.porosity = porosity
        self.heads = np.zeros(numElements, dtype = np.single)
        self.queue = np.zeros(numElements, dtype = np.single)
        self.pointChanges = []

        # The function classes
        self.Groundwater = GroundwaterHandler(self)
        self.Grapher = GraphHandler(plt,self)
        self.Weather = WeatherHandler(self)
        self.Data = DataHandler(self)

    def plotHeads(self):
        self.Grapher.plotHeads()

    def flow(self, iters = 1, track=False, step = 1):
        self.Groundwater.flow(track = track, step = step, iters = iters)

    def __str__(self):
        print("\n")
        print("   ",self.name)
        print(" ","-------------------------")
        print("   ","Conductivity:",self.conductivity, "m/day")
        print("   ","Porosity:",self.porosity)
        print("   ","Specific storage:",self.specificStorage)
        print("   ","Scale:", self.scale, "meters per element")
        print("   ","Time step:",self.timeDelta,"days per time step")
        return ""
