import numpy as np

class Well:

    def __init__(self, xPos, change, tracking = False):
        self.xPos = xPos
        self.change = -abs(change)
        self.total = 0
        if tracking:
            self.pumpOverTime = []
        else:
            self.pumpOverTime = None


class Pump:

    def __init__(self, xPos, change, tracking = False):
        self.xPos = xPos
        self.change = abs(change)
        self.total = 0
        if tracking:
            self.pumpOverTime = []
        else:
            self.pumpOverTime = None

class ModelSnapshot:
    '''
    This class is the container for storing data
    of the flow model at a given time so I don't have to
    worry about where the data is as much
    '''
    def __init__(self, time, data):
        self.time = time
        self.data = data
