import numpy as np

class Well:

    def __init__(self, xPos, change):
        self.xPos = xPos
        self.change = -abs(change)
        self.total = 0


class Pump:

    def __init__(self, xPos, change):
        self.xPos = xPos
        self.change = abs(change)
        self.total = 0
