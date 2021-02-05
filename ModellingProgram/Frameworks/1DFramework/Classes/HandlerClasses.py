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

    def flow(self, iters = 1, time = None, toSteadyState = False):
        if not toSteadyState:
            for i in range(iters):
                self.applyPointChanges()
                vals = self.model.heads[1: ] - self.model.heads[:-1]
                self.model.queue[:-1] += self.model.conductivity*self.model.timeDelta*vals
                self.model.queue[1: ] -= self.model.conductivity*self.model.timeDelta*vals
                self.model.heads += self.model.queue
                self.model.queue = np.zeros(self.model.heads.size, dtype = np.single)

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

class ContaminantHandler:

    def __init__(self):
        pass
