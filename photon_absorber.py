import numpy as np
import matplotlib.pyplot as plt

def updateBoundary(T, T_ext, btype=(1,0,0,0,0,0)):
    """
    Updates the fictional points for the thermal conductor using
    a set of boundary conditions.
    
    The NumPy array is modified in-place.
    
    Arguments:
    ---------
    - T: a 3d array of temperatures within the solid. The shape of the
         array must be (h+1, w+2, d+2) to include the fictional points
         on 5 of the 6 surfaces.
         
    - T_ext: Temperature of the external environment. This may be either
        a scalar or an array. If it is an array, then the external
        temperatures at the 6 surfaces may be specified separately.
        The surfaces must be listed in the order (x0, x1, y0, y1, z0, z1)
        where x0 is the surface with the first index equal to zero, x1 is the
        surface opposite x0, and likewise for the other 4 surfaces.
    
    - btype: Array-like with the boundary types for the 6 surfaces.
        A value of 1 represents perfectly conducting and 0 represents
        perfectly insulating. Any value other than 0 or 1 will yield
        a linear combination of the two types.
        The surfaces are listed in the same order described above.
        
    Returns:
    ------
    Nothing
    
    """
    # Convert T_ext into an array. If it is already an array,
    # no change will occur.
    T_ext = np.ones(6)*T_ext
    
    # For a thermally conductive surface, the fictional points
    # have the temperature of the surroundings. For an insulating
    # surface, the fictional points have the same temperature as the
    # real points that they are in contact with.
    T[0,:,:] = T_ext[0]*btype[0] + T[1,:,:]*(1-btype[0])
    T[-1,:,:] = T_ext[1]*btype[1] + T[-2,:,:]*(1-btype[1])
    T[:,0,:] = T_ext[2]*btype[2] + T[:,1,:]*(1-btype[2])
    T[:,-1,:] = T_ext[3]*btype[3] + T[:,-2,:]*(1-btype[3])
    T[:,:,0] = T_ext[4]*btype[4] + T[:,:,1]*(1-btype[4])
    T[:,:,-1] = T_ext[5]*btype[5] + T[:,:,-2]*(1-btype[5])
    
    # Modifications occur in place so no need to return anything.
    return
    
def thermalStep(T, dx, dt, A):
    """
    Makes one timestep for simulating the diffusion of heat
    through a solid, given an initial temperature distribution,
    assuming a cyclic boundary. If the boundary is not cyclic,
    the caller must independently update the boundaries
    of the returned array.
    
    The array is modified in-place.
    
    Arguments:
    ---------
    - T: a 3D NumPy array containing the initial temperature distribution,
        to be modified in-place.
    
    - dx: the separation in cm between adjacent points in the solid.
    
    - A: the ratio K/(C*rho) where K is the conductivity, C is the specific
        heat capacity, and rho is the density.
    
    - T_ext: Temperature of the external environment. This may be either
        a scalar or an array. If it is an array, then the external
        temperatures at the 6 surfaces may be specified separately.
        The surfaces must be listed in the order (x0, x1, y0, y1, z0, z1)
        where x0 is the surface with the first index equal to zero, x1 is the
        surface opposite x0, and likewise for the other 4 surfaces.
        
    Returns:
    -------
    Nothing
    
    
    """
    # Calculate the sum of adjacent points
    T_sum = np.roll(T, -1, axis=0) + np.roll(T, 1, axis=0) \
          + np.roll(T, -1, axis=1) + np.roll(T, 1, axis=1) \
          + np.roll(T, -1, axis=2) + np.roll(T, 1, axis=2)
    
    # Calculate the change in temperature
    deltaT = A*(T_sum - 6*T)/(dx**2) * dt
    
    # Add the change in temperature to the original array
    T += deltaT
    
    # Nothing to return - the array was modified in-place.
    return
    
    
def thermalSimulate(T, t, substeps=1, dx=0.1, A=1, T_ext=0, btype=(1,0,0,0,0,0)):
    """
    Simulates the diffusion of heat through a solid given an initial
    temperature distribution and boundary conditions. Simulation
    timesteps are saved at specified times. Additional unsaved
    simulation steps may take place between the saved steps if
    specified by the `substeps` parameter.
    
    Note that a perfectly conducting boundary will not in fact
    result in an edge temperature exactly equal to the external
    temperature. The algorithm creates a set of "fictional points"
    just outside the limits of the solid, and enforces boundary
    conditions only on those points.
    
    Arguments:
    ---------
    - T: a 3D NumPy array containing the initial temperature distribution.
         
    - t: an array containing the times to be saved. Should have uniform separation.
        
    - substeps: Number of simulation timesteps for each time interval given
        in the `t` parameter. If more than 1, the times will be subdivided and
        additional simulation steps will be taken (but not saved).
        Default: 1
    
    - dx: the separation in cm between adjacent points in the solid.
        Default: 0.1 cm
    
    - A: the ratio K/(C*rho) where K is the conductivity, C is the specific
        heat capacity, and rho is the density.
        Default: 1
    
    - T_ext: Temperature of the external environment. This may be either
        a scalar or an array. If it is an array, then the external
        temperatures at the 6 surfaces may be specified separately.
        The surfaces must be listed in the order (x0, x1, y0, y1, z0, z1)
        where x0 is the surface with the first index equal to zero, x1 is the
        surface opposite x0, and likewise for the other 4 surfaces.
        Default: 0
        
    - btype: Array-like with the boundary types for the 6 surfaces.
        A value of 1 represents perfectly conducting and 0 represents
        perfectly insulating. Any value other than 0 or 1 will yield
        a linear combination of the two types.
        The surfaces are listed in the same order described above.
        Default: (1, 0, 0, 0, 0, 0)
    
    
    Returns:
    -------
    - simulation: a 4D NumPy array containing only the saved results of the
         simulation, with the first three indices corresponding to
         the spatial coordinates, and the fourth index corresponding
         to the time.
    
    """
    
    # Find the timestep, assuming uniform times
    dt = (t[1]-t[0]) / substeps
    
    # Modify the array to include fictional points
    currentT = np.zeros(np.array(T.shape)+2)
    currentT[1:-1,1:-1,1:-1] = T
    
    # Make an empty array with an extra axis for time
    simulation = np.zeros(T.shape + (len(t),))
    
    # Add the first frame
    simulation[:,:,:,0] = T
    
    # Iterate!
    for step in range(1,len(t)):
        for substep in range(substeps):
            # Update the boundary of our array
            updateBoundary(currentT, T_ext, btype=btype)
            
            # Take a step in the simulation
            thermalStep(currentT, dx, dt, A)
        
        # Add the last substep to the simulation
        simulation[:,:,:,step] = currentT[1:-1,1:-1,1:-1]
    
    return simulation

def getSignature(T, p_loc, sensor_loc=None, p_temp=100., steps=200, dx=0.1, A=1,
                  T_ext=0., substeps=5):
    """
    Simulate a photon absorption event for an absorber and return
    the temperature curve measured by a point sensor located at
    a given position within the absorber. The absorber has a base
    that is in perfect thermal contact with the heat bath, and
    the remaining surfaces are insulated.
    
    Arguments:
    ---------
    - T: An array with the initial temperature distribution for
        the absorber.
    
    - p_loc: Tuple containing the coordinates of the location where
        the photon was absorbed.
        
    - sensor_loc: Tuple with the coordinates of the temperature sensor.
        If none, the location will be at the center of the bottom
        surface (or at the point with indices rounded down, if there
        is no exact center).
        Default: None
    
    - p_temp: The temperature difference caused by the absorbed
        photon.
        Default: 100.0
        
    - steps: Number of temperature measurements.
        Default: 100
    
    - substeps: Number of simulation steps before each temperature
        measurement.
        Default: 5
    
    - dx: The spacing between points in the temperature array.
        Default: 0.1
        
    - A: The ratio K/(C*rho) in units of (T*d^2)/t where K is
        the conductivity, C is the specific heat, and rho is the
        mass density of the absorber.
        Default: 1
        
    - T_ext: The temperature of the external heat bath.
        Default: 0
    
    Returns:
    -------
    - t: An array of times for the temperature measurements
    
    - signature: The measured temperatures 
    
    """
    # Calculate the time between measurements, and
    # construct the time array
    dt = 0.1*dx**2/A * substeps
    t = np.arange(0, dt*steps, dt)
    
    # If the sensor location was not specified, default to
    # the base at the center
    if sensor_loc == None:
        sensor_loc = (0, T.shape[1]//2, T.shape[2]//2)
    
    # Copy the temperature array to avoid overwriting the original
    newT = T.copy()
    newT[p_loc[0], p_loc[1], p_loc[2]] += p_temp
    
    # Simulate!
    sim = thermalSimulate(newT, t, substeps=substeps, T_ext=T_ext, A=A, dx=dx)
    
    # Pull the signature from the simulation data
    signature = sim[sensor_loc[0],sensor_loc[1],sensor_loc[2],:]
    
    return t, signature
    