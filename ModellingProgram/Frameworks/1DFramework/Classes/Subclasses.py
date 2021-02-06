import numpy as np

class Well:

    def __init__(self, model, xPos, change, isActive = False, name = 'unnamed well'):
        self.type = "Well"
        self.name = name
        self.model = model
        self.xPos = xPos
        self.yPos = self.model.Geography.terrain[xPos]
        self.change = -abs(change)
        self.total = 0
        self.isActive = isActive
        self.data = []

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def record(self):
        if self.isActive:
            self.data.append((self.model.timeDelta, self.model.Groundwater.heads[self.xPos]))

class Pump:

    def __init__(self, model, xPos, change, isActive = False):
        self.type = "Pump"
        self.model = model
        self.xPos = xPos
        self.yPos = self.model.Geography.terrain[xPos]
        self.change = abs(change)
        self.total = 0
        self.isActive = isActive
        self.data = []

    def activate(self):
        self.active = True

    def deactivate(self):
        self.isActive = False

    def record(self):
        pass


class QueueChangePoint:
    '''
    Measures the queue change at a single point for a run
    '''
    def __init__(self,xPos,model):
        self.model = model
        self.xPos = xPos
        self.data = []

    def record(self):
        self.data.append(self.model.time,self.model.queue[xPos])


class ModelSnapshot:
    '''
    This class is the container for storing data
    of the flow model at a given time so I don't have to
    worry about where the data is as much
    '''
    def __init__(self, time, data):
        self.time = time
        self.data = data
        self.active = False

    def activate(self):
        self.active = True
