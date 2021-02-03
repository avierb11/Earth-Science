import numpy as np
import matplotlib.pyplot as plt

def _plotHeads(mPlot):
    mPlot.figure(1)
    mPlot.plot(np.linspace(0,self.length,self.numElements),self.heads)
    mPlot.xlim(0,self.length)
    mPlot.xlabel("Length")
    mPlot.ylabel("Hydraulic head")
    mPlot.title("Hydraulic head over the model")
    mPlot.show()

def _plotConc(mPlot,length,numElements):
    mPlot.plot(np.linspace(0,length,self.numElements),self.concentrations)
    mPlot.xlim(0,self.length)
    mPlot.xlabel("X position")
    mPlot.ylabel("Solute Concentration")
    mPlot.title("Solute Concentration in the model")
    mPlot.ylim(0,1)
    mPlot.xlim(0, length)
    mPlot.show()

def _showQueueChanges(mPlot):
    '''Plots a graph of how the queue changes over some iterations'''
    mPlot.plot(mPlot.queueChanges)
    mPlot.xlabel("Iteration")
    mPlot.ylabel("Total Queue")
    mPlot.title("Queue change over time")
    mPlot.show()
