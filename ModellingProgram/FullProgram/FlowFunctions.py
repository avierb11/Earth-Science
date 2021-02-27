"""
This module contains the functions that will be used in the full modelling
program.
"""

import numpy as np
import matplotlib.pyplot as plt


def flow(heads, conductivity, specificYield, timeDelta, scale, variableParameters):
    vals = np.copy(heads[:-1] - heads[1: ])
    queue = np.zeros(heads.size)
    mult = (conductivity * timeDelta) / (specificYield * scale * scale)
    if not variableParameters:
        queue[1: ] += mult*(heads[:-1] - heads[1: ])
        queue[:-1] += mult*(heads[1: ] - heads[:-1])
        heads += queue
    elif variableParameters:
        heads[1: ] += mult[1: ]*heads[1: ]
        heads[:-1] += mult[:-1]*heads[:-1]
    del queue

def addPoint(xLocation, change, moreInfo, constants,scale):
    if xLocation == None or moreInfo == None or change == None or constants == None:
        print("Insufficient data")
    else:
        if (int(xLocation/scale), change, moreInfo) in constants:
            print("Data point already in values")
        else:
            constants.append((int(xLocation/scale), change, moreInfo))

def printData(lst,scale):
    print("xPos\tValue\tOther info")
    for i in range(len(lst)):
        print("{}\t{}\t{}".format(scale*lst[i][0],lst[i][1],lst[i][2]))
