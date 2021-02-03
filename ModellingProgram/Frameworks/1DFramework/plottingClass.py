import matplotlib.pyplot as plt
import numpy as np

class FlowModel:

    def __init__(self, length = 10, numElements = 10, conductivity = .01, diffusivity = .05, dispersivity = .1, timeDelta = 1, porosity = .25):
        self.length = length
        self.numElements = numElements
        self.conductivity = conductivity
        self.heads = np.zeros(numElements)
        self.queue = np.zeros(numElements)
        self.scale = length/numElements
        self.plt = plt
        self.diffusivity = diffusivity
        self.dispersivity = dispersivity
        self.solutePeak = 0   # Accurate tracker for the solute peak
        self.concentrations = np.zeros(numElements)
        self.totalSolute = 0
        self.totalWater = 0
        self.timeDelta = timeDelta
        self.porosity = porosity
        self.pointConstants = []
        self.queueChanges = []
        self.pointConcentrations = []
        self.averageVelocity = 0


    # Flow Functions
    #---------------------------------------------------------------------------
    def flow(self, iters = 1, toSteadyState = False, tolerance = 1e-8):
        '''
        Simple flow function, uses the mod values.
        Also includes flowing to steady state
        '''
        mult = (self.timeDelta*self.conductivity)/self.scale
        #print("mult:",mult)

        if not toSteadyState:
            self.queue[:-1] += mult*(self.heads[1: ] - self.heads[:-1])
            self.queue[1: ] += mult*(self.heads[:-1] - self.heads[1: ])
            #print("queue:",queue)
            self.heads += self.queue
            self.queue = np.zeros(self.numElements)
            #print("New heads:",heads)

        if toSteadyState:
            previous_sum = 1000000
            current_sum = 0
            difference = abs(previous_sum - current_sum)

            while difference > tolerance:
                self.queue[:-1] = mult*(self.heads[1: ] - self.heads[:-1])
                self.queue[1: ] = mult*(self.heads[:-1] - self.heads[1: ])

                previous_sum = current_sum
                current_sum = np.sum((np.abs(self.queue)))
                difference = abs(previous_sum - current_sum)

                self.heads += self.queue
                self.queue = np.zeros(self.numElements)

        self.updateAverageVelocity()


    # Flow related things
    #---------------------------------------------------------------------------
    def averageVelocity(self, single = False, printOnly = False):
        '''
        Determines the average velocity of the water
        '''
        if single:
            avg_q = np.average((self.conductivity/self.scale)*(self.heads[:-1] - self.heads[1: ]))
            avg_v = avg_q / self.porosity

            if printOnly:
                print("Average velocity: {}m/s or {}m/day".format(avg_v,86400*avg_v))
            else:
                return avg_v
        else:
            q = np.zeros(self.numElements - 1)
            q += self.conductivity*(self.heads[:-1] - self.heads[1: ])*(1/self.scale)
            return q/self.porosity

    def updateAverageVelocity(self):
        '''Determines a rough average velocity for various things'''
        self.averageVelocity =  ((self.conductivity/self.scale)*np.average(np.sum(self.heads[:-1] - self.heads[1: ])))/self.porosity

    def flowTime(self, point1 = (0, None), point2 = (0,None), printOnly = False, totalLength = False):
        '''
        Determines the time taken for water to flow a certain distance
        Assumes that velocity is constant throughout the model.
        '''
        if totalLength:
            point1 = (0, None)
            point2 = (self.numElements, None)

        avg_q = np.average((self.conductivity/self.scale)*(self.heads[:-1] - self.heads[1: ]))
        avg_v = avg_q / self.porosity
        distance = abs(point2[0] - point1[0])*self.scale
        time = distance / avg_v
        if printOnly:
            print("Time taken to travel {}m: {} days (seconds?)".format(distance,time))
        else:
            return time


    # Solute Transport
    #---------------------------------------------------------------------------
    def advection(self):
        '''
        Models advection, the transport of solute with the bulk fluid motion
        - Follows the motion of the largest
        '''
        self.solutePeak += self.averageVelocity

        if abs(self.solutePeak) >= self.scale:
            self.solutePeak -= self.scale
            queue = np.zeros(self.numElements)   # Need to determine the direction of flow, because this flows to the right,
            queue[:-1] += (self.concentrations[:-1] - self.concentrations[1: ])  # but it isn't necessarily that direction.
            queue[1: ] += (self.concentrations[1: ] - self.concentrations[:-1])
            self.concentrations += queue
            del queue

    def forceAdvection(self):
        queue = np.copy(self.concentrations[:-1])
        self.concentrations[:-1] -= queue
        self.concentrations[1:] += queue
        del queue

    def diffusion(self, runs = 1):
        for i in range(runs):
            queue = self.concentrations*self.diffusivity*self.timeDelta
            self.concentrations -= 2*queue
            self.concentrations[:-1] += queue[1: ]
            self.concentrations[1: ] += queue[:-1]

    def dispersion(self):
        dispersiveCoefficients = self.dispersivity*self.averageVelocity()
        pass

    def fullSoluteTransport(self):
        self.advection()
        self.diffusion()

    def measurePoint(self, xLocation = None, time = 100, show = False):
        '''
        measures the solute concentration travelling through a single point
        '''
        if not xLocation:
            xLocation = self.numElements - 1
        steps = int(time / self.timeDelta)
        pointConcentrations = np.zeros(steps)
        for i in range(steps):
            pointConcentrations[i] = self.concentrations[xLocation]
            self.fullSoluteTransport()
        if show:
            self.plt.figure(1)
            self.plt.plot(pointConcentrations)
            self.plt.xlabel("Time")
            #self.plt.xlim(0,time)
            self.plt.ylabel("Solute concentration")
            self.plt.title("Solute concentration at point {}m over {} days".format(self.scale*xLocation,time))


    # Plotting things
    #---------------------------------------------------------------------------
    def plotHeads(self):
        self.plt.figure(1)
        self.plt.plot(np.linspace(0,self.length,self.numElements),self.heads)
        self.plt.xlim(0,self.length)
        self.plt.xlabel("Length")
        self.plt.title("Hydraulic head over the model")
        self.plt.ylabel("Hydraulic head")
        self.plt.show()

    def plotConc(self):
        self.plt.figure(1)
        self.plt.plot(np.linspace(0,length,self.numElements),self.concentrations)
        self.plt.xlim(0,self.length)
        self.plt.xlabel("X position")
        self.plt.ylabel("Solute Concentration")
        self.plt.title("Solute Concentration in the model")
        self.plt.ylim(0,1)
        self.plt.xlim(0, length)
        self.plt.show()

    def showQueueChanges(self):
        '''Plots a graph of how the queue changes over some iterations'''
        self.plt.plot(mPlot.queueChanges)
        self.plt.xlabel("Iteration")
        self.plt.ylabel("Total Queue")
        self.plt.title("Queue change over time")
        self.plt.show()

    def showQueueAcceleration(self):
        '''Show a graph of how the queue change changes over time over time'''
        acceleration = np.abs(self.queueChanges[1:]-self.queueChanges[:-1])
        self.plt.plot(acceleration)
        self.plt.xlabel("Iterations")
        self.plt.ylabel("Change in the queue change")
        self.plt.title("Change in the queue change over iterations")
        self.plt.show()


    # Determining some actual useful things
    #---------------------------------------------------------------------------
    def soluteEvolution(self, curves = 5, time = 10):
        '''
        Runs solute transport evolution.
        - Plots a given amount of graphs for solute concentration over a given time.
        - At the moment, it does not contain dispersion
        '''
        plots = []
        step = int(time/curves)
        j = step
        alpha_step = .8/curves
        for i in range(time):
            mod.diffusion()
            mod.forceAdvection()
            if j == step:
                plots.append(np.copy(self.concentrations))
                j = 0
            else:
                j += 1
        for i in range(len(plots)):
            self.plt.plot(np.linspace(0,self.length,self.numElements),plots[i], color = 'blue', alpha = .2 + i*alpha_step)
        self.plt.xlabel("Horizontal position")
        self.plt.ylabel("Solute concentration")
        self.plt.xlim(0,self.length)
        self.plt.ylim(0,1)
        self.plt.title("Solute concentration over time")
        self.plt.show()

    def contaminantTransport(self):
        '''
        Determines the time that a contaminant should take to reach
        a given point.
        '''
        # First, determine the average Darcy velocity
        pass


    # Altering the environment
    #---------------------------------------------------------------------------
    def addWell(self,pos,pumpOut):
        '''Adds a well point change'''
        pass

    def addPump(self,pos,pumpIn):
        '''Adds a pump that pumps water into the ground'''
        pass


    # Weather events
    #---------------------------------------------------------------------------
    def rain(self, amount = .1):
        '''Simulates rain. A constant value of water enters the ground.'''
        pass


    # Printing information and things
    #---------------------------------------------------------------------------
    def __str__(self):
        print("Flow Model")
        print("--------------")
        print("Number of elements:",self.numElements)
        print("Model Length:",self.length, "meters")
        print("Scale", self.scale, ("meters per element"))
        print("Hydraulic conductivity:", self.conductivity)
        return "--------------"

    def checkSums(self):
        if self.totalSolute != self.getSoluteTotal():
            print('Something wrong with solute:')
            print("Expected: ", self.soluteTotal)
            print("Current: ", self.getSoluteTotal())


    # Some ease of use stuff here
    #---------------------------------------------------------------------------
    def setDefaults(self):
        self.heads[0] = 1
        self.heads[-1] = 0
        self.pointConstants.append((0,1))
        self.pointConstants.append((self.numElements-1,0))
        self.pointConcentrations.append((int(self.length/2),1))
        self.concentrations[int(self.length/2)] = 1
        self.totalSolute = np.sum(self.concentrations)
        self.totalWater = np.sum(self.heads)

    def getSoluteTotal(self):
        return np.sum(self.concentrations)

    def getWaterTotal(self):
        return np.sum(self.heads)

    def reset(self):
        self.concentrations = np.zeros(self.numElements)
        self.heads = np.zeros(self.numElements)
        for el in self.pointConstants:
            self.heads[el[0]] = el[1]
        for el in self.pointConcentrations:
            self.concentrations[el[0]] = el[1]
