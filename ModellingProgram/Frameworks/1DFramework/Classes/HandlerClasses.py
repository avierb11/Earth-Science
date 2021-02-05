import numpy as np
from Subclasses import *
import matplotlib.pyplot as plt

'''
In this file, I define the classes that are really just
bins for various function used in the full Model class.

Originally, I wanted to keep all of the functions within a single class,
but that got to be too much. Also, splitting the functionality into different
classes will make things easier when it comes to translating some of the
functions to CUDA. That way, I can just rewrite a class with identically
named functions, but with CUDA applications, and just replace the
class in the FlowModel.
'''

class GroundwaterHandler:
    '''
    To handle the water-related things
    It will not save any of the
    '''
    def __init__(self, model):
        self.model = model

    def flow(self, track = False, step = None, iters = 1, time = None, toSteadyState = False):

        if not toSteadyState:
            for i in range(iters):
                self.applyPointChanges()
                vals = self.model.heads[1: ] - self.model.heads[:-1]
                self.model.queue[:-1] += self.model.conductivity*self.model.timeDelta*vals
                self.model.queue[1: ] -= self.model.conductivity*self.model.timeDelta*vals
                self.model.heads += self.model.queue
                self.model.queue = np.zeros(self.model.heads.size, dtype = np.single)

                if track and (i % step == 0):
                    self.model.Data.addDataPoint()

                self.model.time += self.model.timeDelta

            if track:
                self.model.Data.addDataPoint()

        elif toSteadyState:
            print('Flowing to steady state is not yet implemented')

    def getAverageVelocities(self, scale):
        return (self.model.conductivity/(self.model.porosity*scale))*(self.model.heads[:-1] - self.model.heads[1: ])

    def addPump(self, xPos, change):
        self.model.pointChanges.append(Pump(xPos, change))

    def addWell(self, xPos, change):
        self.model.pointChanges.append(Well(xPos, change))

    def applyPointChanges(self):
        '''
        Apply all of the wells and pumps to the model.
        '''
        for element in self.model.pointChanges:
            self.model.heads[element.xPos] += element.change
            if self.model.heads[element.xPos] < 0:
                self.model.heads[element.xPos] = 0


    def setDefaults(self):
        self.model.pointChanges.append(Pump(0,.1))
        self.model.pointChanges.append(Well(self.model.numElements-1,.1))

class GraphHandler:

    def __init__(self, plt, model):
        self.plt = plt
        self.model = model

    def plotHeads(self):
        self.plt.plot(np.linspace(0,self.model.length,self.model.numElements), self.model.heads, color = 'blue')
        self.plt.xlabel("Length (m)")
        self.plt.ylabel("Hydraulic head (m)")
        self.plt.title("Hydraulic head over the model")
        self.plt.xlim(0, self.model.length)
        self.plt.ylim(0)
        self.plt.show()

    def plotFlowData(self):
        self.plt.figure(1)
        for curve in self.model.Data.flowPoints:
            self.plt.plot(self.model.lengthPoints,curve.data, label = "time: {} days".format(curve.time))
        self.plt.legend()
        self.plt.show()

class ContaminantHandler:

    def __init__(self):
        pass

class WeatherHandler:

    def __init__(self, model):
        self.model = model

    def rain(self, duration = 1, total = .1):
        '''
        Simple rain function. The rain is constant over
        the whole model.
        '''
        steps = int(duration / self.model.timeStep)
        rainPerStep = total / steps

        for i in range(steps):
            self.model.heads += rainPerStep

class DataHandler:

    def __init__(self, model):
        self.model = model
        self.flowPoints = []  # Add a new data class to this each time probably

    def addDataPoint(self):
        self.flowPoints.append(ModelSnapshot(self.model.time, np.copy(self.model.heads)))
