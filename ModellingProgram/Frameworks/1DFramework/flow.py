'''
This file contains the functions dealing with flow.
Functions in this file:
- flow
-
'''
import numpy as np

def _flow(heads,queue,conductivity,scale,timeDelta, iters = 1, toSteadyState = False, tolerance = 1e-8):
    '''
    Simple flow function, uses the mod values.
    Also includes flowing to steady state
    '''
    mult = (timeDelta*conductivity)/scale
    #print("mult:",mult)

    if not toSteadyState:
        queue[:-1] += mult*(heads[1: ] - heads[:-1])
        queue[1: ] += mult*(heads[:-1] - heads[1: ])
        #print("queue:",queue)
        heads += queue
        queue = np.zeros(heads.size)
        #print("New heads:",heads)

    if toSteadyState:
        previous_sum = 1000000
        current_sum = 0
        difference = abs(previous_sum - current_sum)

        while difference > tolerance:
            queue[:-1] = mult*(heads[1: ] - heads[:-1])
            queue[1: ] = mult*(heads[:-1] - heads[1: ])

            previous_sum = current_sum
            current_sum = np.sum((np.abs(queue)))
            difference = abs(previous_sum - current_sum)

            heads += queue
            queue = np.zeros(heads.size)



def _flowTime(mod, point1 = (0,None), point2 = (0,None), printOnly = False, totalLength = False):
    if totalLength:
            point1 = (0, None)
            point2 = (mod.numElements, None)

    avg_q = np.average((mod.conductivity/mod.scale)*(mod.heads[:-1] - mod.heads[1: ]))
    avg_v = avg_q / mod.porosity
    distance = abs(point2[0] - point1[0])*mod.scale
    time = distance / avg_v
    if printOnly:
        print("Time taken to travel {}m: {} days (seconds?)".format(distance,time))
    else:
        return time
