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
        self.length = length
        self.name = name
        self.numElements = numElements
        self.timeDelta = timeDelta
        self.scale = length/numElements

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

    def SetDefaults(self):
        self.Groundwater.addPump(0, .1)
        self.GroundWater.addWell(self.Length,-.1)

    def plotHeads(self):
        self.Grapher.plotHeads()

    def flow(self, iters = 1):
        self.Groundwater.flow(iters)

    def applyPointChanges(self):
        '''
        Apply all of the wells and pumps to the model.
        '''
        pass


    def __str__(self):
        print("\n")
        print("   ",self.name)
        print(" ","-------------------------")
        print("   ","Conductivity:",self.conductivity)
        print("   ","Porosity:",self.porosity)
        print("   ","Specific storage:",self.specificStorage)
        return ""
