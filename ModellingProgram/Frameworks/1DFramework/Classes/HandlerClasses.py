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
    def __init__(self, model, measureWells = False):
        self.model = model
        self.measureWells = measureWells

    def flow(self, track = False, step = None, iters = 1, time = None, toSteadyState = False):

        if not toSteadyState:
            for i in range(iters):

                # Determine whether or not data should be collected fro the run
                recordValues = (track and i % step == 0)

                # Applies points from wells and pumps
                self.preprocessing()
                vals = self.model.heads[1: ] - self.model.heads[:-1]
                self.model.queue[:-1] += self.model.conductivity*self.model.timeDelta*vals
                self.model.queue[1: ] -= self.model.conductivity*self.model.timeDelta*vals
                self.model.heads += self.model.queue

                if recordValues:
                    self.postprocessing()

                self.model.queue = np.zeros(self.model.heads.size, dtype = np.single)

                self.model.time += self.model.timeDelta

            if track:
                self.model.Data.addDataPoint()

        elif toSteadyState:
            print('Flowing to steady state is not yet implemented')

    def getAverageVelocities(self, scale):
        return (self.model.conductivity/(self.model.porosity*scale))*(self.model.heads[:-1] - self.model.heads[1: ])

    # Adding pumps and wells and stuff
    #---------------------------------------------------------------------------
    def addPump(self, xPos, change):
        self.model.pointChanges.append(Pump(self.model, xPos, change))

    def addWell(self, xPos, change):
        self.model.pointChanges.append(Well(self.model, xPos, change))

    # Pre and postproecessing stuff
    #---------------------------------------------------------------------------
    def preprocessing(self):
        self.applyPointChanges()

    def postprocessing(self):
        self.measurePointChanges()
        self.model.Data.addDataPoint()

    def applyPointChanges(self):
        '''
        Apply all of the wells and pumps to the model.
        '''
        for element in self.model.pointChanges:
            self.model.heads[element.xPos] += element.change
            if self.model.heads[element.xPos] < 0:
                self.model.heads[element.xPos] = 0

    def measurePointChanges(self):
        for point in self.model.pointChanges:
            point.record()

    def linearConductivityFit(self, start = .15, end = .05):
        '''
        Changes the hydraulic conductivity to be non-constant
        '''
        self.model.conductivity = np.linspace(start, end, self.model.numElements - 1)

    def setDefaults(self):
        self.model.pointChanges.append(Pump(self.model, 0,.1))
        self.model.pointChanges.append(Well(self.model, self.model.numElements-1,.1))

class GraphHandler:

    def __init__(self, plt, model):
        self.plt = plt
        self.model = model

    def plotHeads(self, other = True):
        self.plt.plot(np.linspace(0,self.model.length,self.model.numElements), self.model.heads, color = 'blue', label = "Hydraulic head")
        if other:
            self.plt.plot(np.linspace(0,self.model.length,self.model.numElements), self.model.Geography.terrain, color = 'black', label = 'Terrain')

            # Adding wells and stuff
            for point in self.model.pointChanges:
                self.plt.scatter(point.xPos, point.yPos, color = 'g')

        self.plt.xlabel("Length (m)")
        self.plt.ylabel("Hydraulic head (m)")
        self.plt.title("Hydraulic head over the model")
        self.plt.xlim(0, self.model.length)
        self.plt.ylim(0)
        plt.legend()
        self.plt.show()

    def plotFlowData(self):
        self.plt.figure(1)
        for curve in self.model.Data.flowPoints:
            self.plt.plot(self.model.lengthPoints,curve.data, label = "time: {} days".format(curve.time))
        self.plt.legend()
        self.plt.title(self.model.name)
        self.plt.xlabel("Location")
        self.plt.ylabel("Hydraulic head")
        self.plt.show()

    def plotWellData(self, wells = 'all'):
        '''
        Plot the data for the water extracted during each run
        at all wells or just a specified well.
        '''
        self.plt.figure(1)
        for well in self.model.pointChanges:
            if well.type == 'Well':
                plt.plot(well.data, label = "Position: {}".format(well.xPos))
        self.plt.legend()
        self.plt.xlabel("Time (unitless right now)")
        self.plt.ylabel("Water removed")
        self.plt.title("Water removed at a given time per well")
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
        self.flowPointsIsActive = True
        self.queueChangePoints = []
        self.queueChangePointsIsActive = True

    def addDataPoint(self):
        self.flowPoints.append(ModelSnapshot(self.model.time, np.copy(self.model.heads)))

    def measureQueueChange(self):
        for point in self.queueChangePoints:
            point.record()

    def addQueueChangePoint(self, xPos):
        self.queueChangePoints.append(QueueChangePoint(xPos, self.model))


class GeographyHandler:

    def __init__(self, model):
        self.model = model
        self.terrain = np.ones(self.model.numElements, dtype = np.single)

    def determineReservoirSize(self):
        '''
        Determines the reservoir sizes
        '''
        # First, determine the local minimums
        pass
