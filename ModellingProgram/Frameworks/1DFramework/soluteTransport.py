import numpy as np

def _advection(mod):
    new = mod.solutePeak + self.averageVelocity()
    if new > int(mod.solutePeak):
        queue = np.zeros(mod.numElements)   # Need to determine the direction of flow, because this flows to the right,
        queue[:-1] += (mod.concentrations[:-1] - mod.concentrations[1: ])  # but it isn't necessarily that direction.
        queue[1: ] += (mod.concentrations[1: ] - mod.concentrations[:-1])
        mod.concentrations += queue
        del queue

def _forceAdvection(mod):
    queue = np.copy(mod.concentrations[:-1])
    mod.concentrations[:-1] -= queue
    mod.concentrations[1:] += queue
    del queue

def _diffusion(mod):
    for i in range(runs):
        queue = self.concentrations*self.diffusivity*self.timeDelta
        mod.concentrations -= 2*queue
        mod.concentrations[:-1] += queue[1: ]
        mod.concentrations[1: ] += queue[:-1]
