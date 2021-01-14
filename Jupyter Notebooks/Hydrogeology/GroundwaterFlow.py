
def calculateHydraulicHead(z, P):
    '''
    Calculates the hydraulic head for a given elevation and pressure.
    '''
    return z + P/(1000 * 9.81)

def calculatePressureHead(h, z):
    '''
    Calculates the pressure head for a given hydraulic
    head and elevation head
    '''
    return (9810)*(h - z)



def dischargeDarcy(K, length, width, h1, h2, L):
    '''
    returns the discharge. 
    -K is hydraulic conductivity
    -Length and width are the dimensions of the flow area
    -h1 and h2 are the hydraulic head values
    -L is the distance between the two points
    '''
    return K*A*(h2-h1)/L

def specificDischarge(K, h1, h2, L):
    '''
    Returns the specific discharge (Darcy velocity)
    '''
    return (K/L)*(h2-h1)

def avgVelocity(q, theta):
    '''
    Returns the average velocity of flow given a Darcy velocity and porosity
    '''
    return q/theta


def getQueue(heads):
    '''
    Calculates the queue for a given head array.
    It assumes that the boundaries of the aquifer are impermeable.
    '''
    # First, define the queue
    queue = np.zeros(heads.shape)
    
    # Determine the difference in all 4 directions
    queue[:-1,:] += heads[1: ,:] - heads[:-1,:]   # Top side
    queue[1: ,:] += heads[:-1,:] - heads[1: ,:]   # Bottom side
    queue[:,:-1] += heads[:,1: ] - heads[:,:-1]   # Left side
    queue[:,1: ] += heads[:,:-1] - heads[:,1: ]   # Right side
    
    return queue

def discharge2D(heads, conductivity, heights, scaleX = 1, scaleY = 1):
    '''
    Determines the net discharge for each point on the grid.
    -Uses the heads at a the points on the grid. to determine dh
    -Use a scaling factor that relates the dimensions of the blocks in either
        direction to calculate dL
    -This function is more general and uses heterogeneous (but isotropic) conductivity,
        as well as grid spaces with different sizes.
    -Requires an array of flow area heights
    '''
    
    # Calculate the queue
    queue = np.zeros(heads.shape)
    queue[:-1,:] += (heads[1: ,:] - heads[:-1,:])*conductivity[:-1,:]*heights[:-1,:]*(1/scaleY)  #need to multiply by area
    queue[1: ,:] += (heads[:-1,:] - heads[1: ,:])*conductivity[1: ,:]*heights[1: ,:]*(1/scaleY)
    queue[:,:-1] += (heads[:,1: ] - heads[:,:-1])*conductivity[:,:-1]*heights[:,:-1]*(1/scaleX)
    queue[:,1: ] += (heads[:,:-1] - heads[:,1: ])*conductivity[:,1: ]*heights[:,1: ]*(1/scaleX)
    
    # Update heads
    heads += queue
    
    # Delete queue
    del queue
    
    return heads



def updateHeads2D(heads, conductivity, heights, Ss, timeStep = 1, scaleX = 1, scaleY = 1):
    '''
    Determines the net discharge for each point on the grid.
    -Uses the heads at a the points on the grid. to determine dh
    -Use a scaling factor that relates the dimensions of the blocks in either
        direction to calculate dL
    -Requires specific storage array (Ss)
    -This function is more general and uses heterogeneous (but isotropic) conductivity,
        as well as grid spaces with different sizes.
    -Requires an array of flow area heights
    -Requires a time step: smaller will be more accurate
    -Returns the heads array fully updated
    '''
    
    # Calculate the queue
    queue = np.zeros(heads.shape)
    queue[:-1,:] += (heads[1: ,:] - heads[:-1,:])*conductivity[:-1,:]*heights[:-1,:]*(1/scaleY)*timeStep
    queue[1: ,:] += (heads[:-1,:] - heads[1: ,:])*conductivity[1: ,:]*heights[1: ,:]*(1/scaleY)*timeStep
    queue[:,:-1] += (heads[:,1: ] - heads[:,:-1])*conductivity[:,:-1]*heights[:,:-1]*(1/scaleX)*timeStep
    queue[:,1: ] += (heads[:,:-1] - heads[:,1: ])*conductivity[:,1: ]*heights[:,1: ]*(1/scaleX)*timeStep
    # Currently, we have the net volume of water change for each element
    
    # Convert it now to dH
    queue = queue/(Ss*heights*scaleX*scaleY)
    
    # Update heads
    heads += queue
    
    # Delete queue
    del queue
    
    return heads





