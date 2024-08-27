# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:12:07 2023

@author: Collin Sutton
prepared for "A laboratory-validated, graph-based flow and transport model for naturally fractured media" 
submitted to Geophysical Research Letters
Reuse of this code must cite the original paper
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import random
import time
import pandas as pd
from scipy import linalg as LA
from scipy import integrate
from sklearn.metrics.pairwise import euclidean_distances
import bisect
from collections import Counter 
import os

os.chdir("c:\\\\Users\\colli\\Documents\\Python Scripts")
plt.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 18})

def r_fun(mu, dx, dy, bi, bj):
    r = (6*mu*(dx/dy)*((1/bi**3)+(1/bj**3)))
    return r

def quantile_calc(btc_1d, timearray, quantile):
    # calculate cumulative amount of solute passing by location
    M0i = integrate.cumtrapz(btc_1d, timearray)
    # normalize by total to get CDF
    quant = M0i/M0i[-1]
    # calculate midtimes
    mid_time = (timearray[1:] + timearray[:-1]) / 2.0
    
    # now linearly interpolate to find quantile
    gind = np.argmax(quant > quantile)
    m = (quant[gind] - quant[gind-1])/(mid_time[gind] - mid_time[gind-1])
    b = quant[gind-1] - m*mid_time[gind-1]
    
    tau = (quantile-b)/m
    return tau

def inputExpData(filename, loopTime):
    file = pd.read_csv(filename) #data file contain raw C output data of valid experiments
    col_names = file.columns
    print("column names:", col_names)
    
    # Replace NaN values in df with 0
    file[:] = file[:].fillna(0)

    # manually assign value to variable
    trad2 = file["time"].to_numpy()
    trad2 = trad2[:loopTime]
    trad3 = file["time"].to_numpy()
    granPoint5_rad2 = file["rad2"].to_numpy()
    granPoint5_rad2 = granPoint5_rad2[:loopTime]
    granPoint5_rad3 = file["rad3"].to_numpy()
    granpoint5_rad2RTD = granPoint5_rad2/np.trapz(granPoint5_rad2) #RTD equation 16 https://doi.org/10.1016/j.ces.2018.11.001
    granpoint5_rad3RTD = granPoint5_rad3/np.trapz(granPoint5_rad2) #RTD equation 16 https://doi.org/10.1016/j.ces.2018.11.001
    granpoint5_rad2Norm = granPoint5_rad2/np.max(granPoint5_rad2) #
    granpoint5_rad3Norm = granPoint5_rad3/np.max(granPoint5_rad2) #
    
    return(trad2,trad3, granPoint5_rad2,granPoint5_rad3,granpoint5_rad2RTD,granpoint5_rad3RTD,granpoint5_rad2Norm,granpoint5_rad3Norm)


def graphModel(trad2, trad3,granPoint5_rad2, granPoint5_rad3, granpoint5_rad2RTD, granpoint5_rad3RTD, granpoint5_rad2Norm, granpoint5_rad3Norm, graph_filename, dy, alpha, flowrate, model_run_time, alphaRW):
    
    
    
    graphFull = np.loadtxt(graph_filename + '.txt', delimiter=' ')
    
    
    
    
    tic = time.perf_counter()  
    ## Fracture properties
    n = len(graphFull[:,0])
    core_width = 0.05 #[m] - 5cm
    inlet_loc = 0
    outlet_loc = 0.1
    # fixed channel width
    dy = dy #this one looks great with alpha = 0.2
    mu = 8.9E-4 #[Pa*s]
    
    ##use PET data for graph
    x = graphFull[:,0] /100 # voxel size [cm] to m
    y = graphFull[:,1] /100 # voxel size [cm] to m
    z = graphFull[:,2] /100 # voxel size [cm] to m
    clusterID = graphFull[:,3]
    aperature = graphFull[:,4]  # max concentration
    
    calcAp = np.zeros_like(aperature)
    ##need to turn "aperature" from C max into a value for b -- must be done here so that inlets and outlets can be set after
    alpha = 0.7
    for i in range(len(aperature)):
        calcAp[i] = (aperature[i]*(0.07763/100))/alpha
    aperature = calcAp

    #set boundary locations
    inlet_loc = -0.0001
    outlet_loc = 0.095 # set to length of core
    inlet1 = np.array([max(x)/2,max(y)/2, inlet_loc])
    
    outlet1 = np.array([max(x)/2,max(y)/2, outlet_loc])
    outlet2 = np.array([core_width,0.02876, outlet_loc]) #this moves the outlet to match the curve of the fracture plane
    outlet3 = np.array([0,max(y)/2, outlet_loc])
    
    points = np.column_stack((x,y,z))
    
    ##add inlet and outletc conditions
    ## must add the appropriate number of outlets and inlets to points and aperature arrays
    points = np.append(points, [inlet1, outlet1, outlet2, outlet3],axis = 0)
    clusterID = np.append(clusterID, [-1,-1], axis = 0)
    aperature = np.append(aperature, [np.mean(aperature),np.mean(aperature),np.mean(aperature), np.mean(aperature)], axis = 0) #this is 1/32" ID tubing in mm
    pointsSorted = sorted(points , key=lambda k: [k[2], k[0]]) #this sorts the points from low x to greatest x
    pointsSorted = np.array(pointsSorted)
    distances = euclidean_distances(pointsSorted)
    # set distances of zero equal to very big number. This makes it so that you aren't connecting a point to itself
    distances[distances==0]=1e26
  
    ## Transport properties
    ## diffusion coefficient
    Dm = 6.7E-10
    
    ###Set boundary conditions for known pressure and flowrate###
    flowrate = flowrate
    
    #How many inlets do you have?
    inlets = 1 #this is for one inlet (like a core)
    n = n + inlets
    #make an array for particle tracking
    inletArray = np.zeros((inlets,1))
    for i in range(inlets):
        inletArray[i] = i
        
    #How many outlets do you have?
    outlets = 3 # this is for 3 outlets

    #How many known pressures do you have?
    numBounds = 3 #One known pressure - the outlet
    n = n + numBounds
    
    #set the outlet pressure conditions
    outlet_pressure = 101325 #[Pa] - represents tubing at core during flow-through experiments
    #set outlet pressures if different pressures
    outlet_pressures = np.asarray((101325,101325,101325))
    
    #find that nodes of known pressures - example below for a core
    ##find inlet and outlet index if using flow-through experiment
    # inlet_ind = int(np.argwhere(pointsSorted[:,2]== np.min(pointsSorted[:,2])))
    # outlet_ind = int(np.argwhere(pointsSorted[:,2]== np.max(pointsSorted[:,2])))
    
    # #this is an example of only using one known pressure node
    # m = [[outlet_ind,outlet_pressure]]
    # m = np.array(m,dtype=object)
    
    #This is an example of using multiple known pressure nodes
    outlet_ind = np.argwhere(pointsSorted[:,2]== np.max(pointsSorted[:,2]))
    
    outletArray = np.zeros((outlets,2))
    for i in range(outlets):
        outletArray[i,0] = outlet_ind[i]
        outletArray[i,1] = outlet_pressures[i]
    #How many known pressures do you have?
    
    ##make the adjacency matrix 
    k = 4 #number of distances to extract
    edges = np.zeros((n,k), dtype=int)
    # preallocate adjacency matrix
    A = np.zeros((n,n))
    # preallocate flow connection matrix
    Q = np.zeros((n,n))
    b = np.zeros((n,1))
    PV = np.zeros((n,n))

    # preallocate edges dictionary
    edges = {}

    #This statement takes the index of the smallest k number of values from each row 
    #Read as shortest 4 distances from a node to other nodes (one of the 4 is itself)
    #Makes weighted adjacency matrix
    for i in range(n):
        ind = np.argpartition(distances[i,:], k)[:k] #this is grabbing the shortest 4 distances from a node to other nodes (one of the 4 is itself)
        aval = 1/(r_fun(mu, distances[i,ind], dy, aperature[i], aperature[ind]))
        A[i,ind] = aval
        A[ind, i]= aval
        edges[i] =  ind.astype(int)
        # PV[i,ind] = distances[i,ind]*((aperature[i]+aperature[ind])/2) *dy # [m^3]
            
    # Check to makes sure all edge connections are mutual
    for i in range(n):    
        for j in edges[i]:
            if i not in edges[j]:
                edges[j] = np.append(edges[j], i)
                print('Edge: ' + str(j) + ' to '+ str(i) + ' added')
    
    ###had to add this after making connections mutual or it wouldnt work properly    
    for i in range(n):    
        for j in edges[i]:
            PV[i,j] = distances[i,j]*((aperature[i]+aperature[j])/2) *dy # [m^3]
    
    #Make diagonal matrix [D] from adjacency matrix [A]
    d_center = np.sum(A, axis=1)
    D = np.diag(d_center)
    
    
    #create Laplacian matrix [L]
    L = D-A
    
    #sort so that early known nodes are handled first  
    mSorted = outletArray[outletArray[:,0].argsort()]
    mSortedLoop = mSorted #this changes the index of the node to account for deleting rows of L matrix
    #add pressure boundary conditions to b vector
    for i in range(numBounds):
        for j in edges[mSorted[i,0]]:
            b[j] = b[j] + mSorted[i,1] * A[np.int64(mSorted[i,0]),j] #set the b matrix to known pressure / R at node and connections
            #should this be negative????
    
    #remove known pressure nodes from the graph matrix and b vector
    for i in range(numBounds): ############ TRY TO VECTORIZE
        L = np.delete(L, np.int64(mSortedLoop[i,0]), 0) #delete row in L matrix corresponding to boundary pressure condition node
        L = np.delete(L, np.int64(mSortedLoop[i,0]), 1) #delete column in L matrix corresponding to boundary pressure condition node
        b = np.delete(b, np.int64(mSortedLoop[i,0]), 0) #delete row in b vector corresponding to boundary pressure condition node
        mSortedLoop-= 1    
    
    ###Add flowrate information to boundary condition vector (b)###
    #find that nodes of known flowrate - example below for a core
    ##find inlet index if using flow-through experiment
    inlet_ind = np.zeros((inlets,1))
    inlet_ind = np.int64(np.argwhere(pointsSorted[:,2]== np.min(pointsSorted[:,2])))
    
    b[inlet_ind] = flowrate
    
    
    ##solve for unknown pressure heads
    h = LA.solve(L,b)
    
    check = L@h - b
    print('Residual numerical error: %e' %np.sum(check))
    
    #add know pressures back into h vector
    for i in range(len(outletArray)):
        h = np.insert(h, np.int64(outletArray[i,0]), outletArray[i,1])
    
    # preallocate transport related matrices
    V = np.zeros((n,n))
    P = np.zeros((n,n))
    T1 = np.zeros((n,n))
    T2 = np.zeros((n,n))
    tdisc = 1000
    F = np.zeros((n,n, tdisc-1))
    Re = np.zeros((n,n))
    PSI = np.zeros((n,n))
    
        
    # # Calculate flow through each edge using peclet number
    # for node_i, js in edges.items():
    #     # print(node_i)
    #     # print(js)
    #     for node_j in js:
    #         # calculate pressure drop
    #         dp = h[node_i] - h[node_j]
    #         # calculate flow
    #         Q[node_i, node_j] = dp/r_fun(mu, distances[node_i, node_j], dy, aperature[node_i], aperature[node_j])
    #         # Divide flow by cross-sectional area to calculate velocity between nodes
    #         V[node_i, node_j] = Q[node_i, node_j]/(dy*(aperature[node_i]+ aperature[node_j])/2)
    #         # equation 12 a
    #         if abs(V[node_i, node_j]) > 0:
    #             T1[node_i, node_j] = distances[node_i, node_j]/abs(V[node_i, node_j])
    #             # equation 12 b
    #             T2[node_i, node_j] = distances[node_i, node_j]**2/Dm
    #             # calculate peclet number
    #             peclet = distances[node_i, node_j]*abs(V[node_i, node_j]) /Dm
    #             print(peclet)
    #             if V[node_i, node_j] > 0:
    #                 # Equation 13a
    #                 P[node_i, node_j] = peclet/(1 - np.exp(-peclet))
    #             elif V[node_i, node_j] < 0:
    #                 # Equation 13b
    #                 P[node_i, node_j] = peclet/(np.exp(peclet) - 1)
    
    
    # Calculate flow through each edge using large peclet number          
    for node_i, js in edges.items():
        # print(node_i)
        # print(js)
        for node_j in js:
            # calculate pressure drop
            dp = h[node_i] - h[node_j]
            # calculate flow
            Q[node_i, node_j] = dp/r_fun(mu, distances[node_i, node_j], dy, aperature[node_i], aperature[node_j])
            # Divide flow by cross-sectional area to calculate velocity between nodes
            V[node_i, node_j] = Q[node_i, node_j]/(dy*(aperature[node_i]+ aperature[node_j])/2)
            # equation 12 a
            if abs(V[node_i, node_j]) > 0:
                T1[node_i, node_j] = distances[node_i, node_j]/abs(V[node_i, node_j])
                # equation 12 b
                T2[node_i, node_j] = distances[node_i, node_j]**2/Dm
                # calculate peclet number
                peclet = distances[node_i, node_j]*abs(V[node_i, node_j]) /Dm
                # print(peclet)
                if V[node_i, node_j] > 0:
                    # Equation 13a
                    if peclet > 800:
                        P[node_i, node_j] = peclet
                    else:
                        P[node_i, node_j] = peclet/(1 - np.exp(-peclet))
                elif V[node_i, node_j] < 0:
                    # Equation 13b
                    if peclet > 800:
                        P[node_i, node_j] = 0
                    else:
                        P[node_i, node_j] = peclet/(np.exp(peclet) - 1)
                    
    # # Calculate flow through each edge using Reynold's Number
    # for node_i, js in edges.items():
    #     # print(node_i)
    #     # print(js)
    #     for node_j in js:
    #         # calculate pressure drop
    #         dp = h[node_i] - h[node_j]
    #         # calculate flow
    #         Q[node_i, node_j] = dp/r_fun(mu, distances[node_i, node_j], dy, aperature[node_i], aperature[node_j])
    #         # Divide flow by cross-sectional area to calculate velocity between nodes
    #         V[node_i, node_j] = Q[node_i, node_j]/(dy*(aperature[node_i]+ aperature[node_j])/2)
    #         Re[node_i, node_j] = ((1000*V[node_i, node_j]*2*((aperature[node_i]+ aperature[node_j])/2))/(mu))
    #         # equation 12 a
    #         if abs(V[node_i, node_j]) > 0:
    #             T1[node_i, node_j] = distances[node_i, node_j]/abs(V[node_i, node_j])
    #             # equation 12 b
    #             T2[node_i, node_j] = distances[node_i, node_j]**2/Dm
    #             # calculate reynolds number
    #             re = ((1000*abs(V[node_i, node_j])*(2*((aperature[node_i]+ aperature[node_j])/2)))/(mu))
    #             print(re)
    #             if V[node_i, node_j] > 0:
    #                 # Equation 13a
    #                 P[node_i, node_j] = re/(1 - np.exp(-re))
    #             elif V[node_i, node_j] < 0:
    #                 # Equation 13b
    #                 P[node_i, node_j] = re/(np.exp(re) - 1)
    
    
    # Plot V matrix
    plt.figure(figsize=(5.5,4),dpi=150)    
    plt.pcolor(V*60*100, cmap='RdBu')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Advection matrix (V) [cm/min]')
    plt.show()
    
    # plot the graph
    plt.figure(figsize=(7,3),dpi=150)
    # interpolate pressure field
    triang = tri.Triangulation(pointsSorted[:,2], pointsSorted[:,0])
    cf = plt.tricontourf(pointsSorted[:,2], pointsSorted[:,0], np.squeeze(h/1000), 100, cmap="Blues")
    plt.colorbar(cf, format='%.4f', label='kPa')
    
    # plot connections
    qmax = np.max(Q)
    # loop through rows of the Q matrxi
    for i in range(n):
        for j in range(i, n, 1):
            if abs(Q[i,j]) > 0:
                plt.plot([pointsSorted[i,2], pointsSorted[j,2]], [pointsSorted[i,0], pointsSorted[j,0]], color='k', linewidth = 3*abs(Q[i,j])/qmax)
    
    # Plot points based on aperature size
    amax = np.max(aperature)
    plt.scatter(pointsSorted[:,2], pointsSorted[:,0], s=aperature/amax*100, c='k')
    
    # inlet point
    plt.plot(pointsSorted[inlet_ind,2], pointsSorted[inlet_ind,0], 'og')
    # plot outlet point
    plt.plot( pointsSorted[outlet_ind,2], pointsSorted[outlet_ind,0], 'or')
    
    # plt.title('Interpolated pressure field along graph (%d nodes)' % n)
    # plt.xlim([])
    plt.xlabel('meters')
    plt.ylabel('meters')
    plt.tight_layout()
    plt.show()
    
    
    ### Now start calculating transport
    # calculate matrix with probability of spatial displacement
    Psum = np.sum(P,axis=1)
    G = 1/Psum
    for i in range(outlets):     #this looks for outlet indexes and then sets the row to 0
        P[np.int64(outletArray[i,0]),:] = 0
        G[np.int64(outletArray[i,0])] = 1
    P = np.multiply(P, G[:, np.newaxis])
    # calculate cumulative distribution for each node (corresponding to each row in the following)
    Pij = np.cumsum(P, axis=1)
    
    ## Plot p matrix
    plt.figure(figsize=(5.5,4),dpi=150)    
    plt.pcolor(Pij, cmap='PuRd')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Cummulative probability connection matrix (pij)')
    plt.show()
    
    
    # # # Calculate transition time distribution 
    # t = np.logspace(-1, np.log10(T1.max())*10, tdisc)
    # tplot = t[:-1]+np.diff(t)/2

    # # choose random uniform variable to figure out what node the particle will go to next
    # a1 = random.uniform(0, 1)
    # current_i = 1
    # chosen_j = (a1<Pij[current_i,:]).argmax()
    # # find corresponding time for particle
    # a2 = random.uniform(0, 1)
    # # method 1
    # # t_ind = (a2<F[current_i,chosen_j,:]).argmax()
    # # method 2
    # t_ij = np.interp(a2, F[current_i,chosen_j,:], tplot)
    # print(t_ij)
    
    # Now from the beginning
    # define times when particles are going to recorded (seconds)
    model_run_time = model_run_time
    timestep = 60
    T = np.arange(0, model_run_time+1, timestep)
    # time of particle injection
    injection_time = len(trad2) #this corresponds with the experimental injection time
    # number of particles
    zn = 100000
    try: # Try to use radiation data to define inlet pulse. If not defined then use square pulse
        ####using "trad4" if skipping this section since it doesnt exist - ---- actual name is "trad2" if trying to use rad 2 btc 
        injection_time = len(trad2) #this corresponds with the experimental injection time
        #make a CDF to sample from for injection of particles based on injection of FDG - this accounts for tubing dispersion
        cdf = np.cumsum(granpoint5_rad2RTD) 
        release_t = np.zeros((zn,1))
    
        for i in range(zn):
            a3 = random.uniform(0, 1)
            cdfInd = bisect.bisect_left(cdf, a3)
            release_t[i] = trad2[cdfInd]
    
        fig, ax1 = plt.subplots()
        ax1.plot(trad2, granpoint5_rad2RTD, color = "blue")
        ax2 = ax1.twinx()
        ax2.plot(trad2, cdf, color = "black")
        plt.xlabel('Time [second]')
        plt.show()
        
    except NameError:
        # time of particle injection
        injection_time = 10
        # release time of each particle, evenly space particle injections
        release_t = np.linspace(0, injection_time, zn)
    
    # Preallocate particle location matrix Z[x,y,z]
    Z = np.zeros((zn,3,T.size))
    # if inlet_ind.size > 1:
    if inletArray.size > 1:
        # calculate flux weighted particle distribution
        Q_inlet_nodes = np.sum(np.squeeze(Q[inlet_ind,:]), axis=1)
        # particle node distribution
        pnode_dist = np.round(Q_inlet_nodes/np.sum(Q_inlet_nodes)*zn)
        pnode_dist = pnode_dist.astype(int)
        # based on the flux weighted distribution assign particle starting locations
        ind_start = 0
        for i in range(inletArray.size):
            Z[ind_start: ind_start+pnode_dist[i], 0, 0] = pointsSorted[inlet_ind[i],0]
            Z[ind_start: ind_start+pnode_dist[i], 1, 0] = pointsSorted[inlet_ind[i],1]
            Z[ind_start: ind_start+pnode_dist[i], 2, 0] = pointsSorted[inlet_ind[i],2]
            ind_start += pnode_dist[i]
            if i == np.max(inletArray):
                if zn - ind_start != 0:
                    pnode_count = zn - ind_start
                    Z[ind_start: ind_start+pnode_count, 0, 0] = pointsSorted[inlet_ind[i],0]
                    Z[ind_start: ind_start+pnode_count, 1, 0] = pointsSorted[inlet_ind[i],1]
                    Z[ind_start: ind_start+pnode_count, 2, 0] = pointsSorted[inlet_ind[i],2]
            
    # if there is only one injection node then assign all of the initial particle locations to that node
    elif inletArray.size ==1:    
        # pnode_dist = np.array([zn])
        Z[:,0,0] = pointsSorted[inlet_ind,0]
        Z[:,1,0] = pointsSorted[inlet_ind,1]
        Z[:,2,0] = pointsSorted[inlet_ind,2]
    
    # Set up structure to also record breakthrough curves
    # This structure will record the breakthrough time of each particle at outlet node x
    Btc = np.zeros((zn, np.int64(outletArray.size/2))) # divide outletArray.size by 2 to account for the second dimension
    
    #Each particle in Z is at the inlet
    #first step is to sample from the Pij to determine which node a particle will move to
 
    #### Random Walk Particle Tacking #####
    alphaRW = alphaRW
    # particleTotalVelocity = np.zeros((zn,1))
    particleAvgVelocity = np.zeros((zn,1))
    particleAvgVelocity_distWeighted = np.zeros((zn,1))
    particleTimeVelocity = np.zeros((zn,len(T)))
        
    for i in range(zn): #zn
        periodTransTime = 0 + release_t[i]
        try: # Try to use radiation data to define inlet pulse. If not defined then use square pulse
            inletParNode = list(Counter(np.where(Z[i,:,0] == pointsSorted[inlet_ind])[0]).values()).index(len(pointsSorted[inlet_ind]))
        
        except ValueError:
            inletParNode = inlet_ind # only works for 1 inlet
            inletParNode = inletParNode.item()
        currentNode = inletParNode
        particleTotalVelocity = 0
        particleTotalVelocity_distWeighted = 0
        particleVelCount = 0
        particletotalDistance = 0
    
        #This loop takes a particles through the time length of the experiment based on a set timestep (could be a minute - could be 30)
        for t in range(len(T)):
            #the while loop checks a variable (periodTransTime) to see if the particle has exceeded the timestep, if it should keep moving, or if it has left the outlet
            while release_t[i] >= timestep*t:
                Z[i,0,t] = Z[i,0,0] # x coord of the node for transition with interpolation for where the particle is if not at node
                Z[i,1,t] = Z[i,1,0] # y coord of the node for transition
                Z[i,2,t] = Z[i,2,0] # z coord of the node for transition
                t+=1
            
            while periodTransTime <= T[t]:
                a1 = random.uniform(1E-3, 1) # random number for probability of node selection
                a2 = random.uniform(-1, 1)
                chosen_j = np.searchsorted(Pij[currentNode,:], a1)
                particleVelCount += 1
                if chosen_j in outletArray[:,0]:
                    D = alphaRW * V[currentNode, chosen_j] + Dm
                    if a2 >0:
                        znTransTime = (np.sqrt(D**2*a2**4+ 2*D*distances[currentNode, chosen_j]*V[currentNode, chosen_j]*a2**2) + D*a2**2 + distances[currentNode, chosen_j]*V[currentNode, chosen_j])/V[currentNode, chosen_j]**2
    
                    else:
                        znTransTime = (- np.sqrt(D**2*a2**4+ 2*D*distances[currentNode, chosen_j]*V[currentNode, chosen_j]*a2**2) + D*a2**2 + distances[currentNode, chosen_j]*V[currentNode, chosen_j])/V[currentNode, chosen_j]**2
    
                    periodTransTime += znTransTime
                    velocityInterp = distances[currentNode, chosen_j]/znTransTime
                    particleTotalVelocity = particleTotalVelocity + velocityInterp
                    particleTotalVelocity_distWeighted = particleTotalVelocity_distWeighted + (velocityInterp*distances[currentNode, chosen_j])
                    particletotalDistance = particletotalDistance + distances[currentNode, chosen_j]
                    
                    particleAvgVelocity[i] = particleTotalVelocity / particleVelCount
                    particleAvgVelocity_distWeighted[i] = particleTotalVelocity_distWeighted / particletotalDistance
                    particleTimeVelocity[i,t] = velocityInterp
                    
                    btcOutletInd = np.where(outletArray[:,0]==chosen_j)
                    Btc[i, btcOutletInd] = periodTransTime
                    Z[i,0,t:] = pointsSorted[np.int64(outletArray[btcOutletInd,0]),0]
                    Z[i,1,t:] = pointsSorted[np.int64(outletArray[btcOutletInd,0]),1]
                    Z[i,2,t:] = pointsSorted[np.int64(outletArray[btcOutletInd,0]),2]
                    if periodTransTime >= model_run_time:
                        timeDif = periodTransTime - T[t]
                        interpPos = timeDif * distances[currentNode, chosen_j]/znTransTime

                        particleTimeVelocity[i,t] = velocityInterp
                        d = (((pointsSorted[currentNode, 0] - pointsSorted[chosen_j,0])**2)+((pointsSorted[currentNode, 1] - pointsSorted[chosen_j,1])**2)+((pointsSorted[currentNode, 2] - pointsSorted[chosen_j,2])**2))**0.5
                        xNew = pointsSorted[chosen_j, 0] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,0]-pointsSorted[currentNode, 0]))
                        yNew = pointsSorted[chosen_j, 1] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,1]-pointsSorted[currentNode, 1]))
                        zNew = pointsSorted[chosen_j, 2] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,2]-pointsSorted[currentNode, 2]))
                        Z[i,0,t:] = xNew.item() # x coord of the node for transition with interpolation for where the particle is if not at node
                        Z[i,1,t:] = yNew.item() # y coord of the node for transition
                        Z[i,2,t:] = zNew.item() # z coord of the node for transition
                        break
                    elif periodTransTime >= T[t]:
                        timeDif = periodTransTime - T[t]
                        if timeDif >= timestep:
                            while timeDif >= timestep:
                                interpPos = timeDif * distances[currentNode, chosen_j]/znTransTime
                                
                                particleVelCount += 1

                                d = (((pointsSorted[currentNode, 0] - pointsSorted[chosen_j,0])**2)+((pointsSorted[currentNode, 1] - pointsSorted[chosen_j,1])**2)+((pointsSorted[currentNode, 2] - pointsSorted[chosen_j,2])**2))**0.5
                                xNew = pointsSorted[chosen_j, 0] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,0]-pointsSorted[currentNode, 0]))
                                yNew = pointsSorted[chosen_j, 1] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,1]-pointsSorted[currentNode, 1]))
                                zNew = pointsSorted[chosen_j, 2] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,2]-pointsSorted[currentNode, 2]))
                                
                                try:
                                    Z[i,0,t] = xNew.item() # x coord of the node for transition with interpolation for where the particle is if not at node
                                    Z[i,1,t] = yNew.item() # y coord of the node for transition
                                    Z[i,2,t] = zNew.item() # z coord of the node for transition
                                    particleTimeVelocity[i,t] = velocityInterp
                                except IndexError:
                                    t-=1
                                    Z[i,0,t:] = Z[i,0,t] # x coord of the node for transition with interpolation for where the particle is if not at node
                                    Z[i,1,t:] = Z[i,0,t] # y coord of the node for transition
                                    Z[i,2,t:] = Z[i,0,t] # z coord of the node for transition
                                    particleTimeVelocity[i,t] = velocityInterp
                                    break
                                t+=1
                                timeDif = timeDif - timestep
    
                        interpPos = timeDif * distances[currentNode, chosen_j]/znTransTime

                        particleTimeVelocity[i,t] = velocityInterp
                        d = (((pointsSorted[currentNode, 0] - pointsSorted[chosen_j,0])**2)+((pointsSorted[currentNode, 1] - pointsSorted[chosen_j,1])**2)+((pointsSorted[currentNode, 2] - pointsSorted[chosen_j,2])**2))**0.5
                        xNew = pointsSorted[chosen_j, 0] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,0]-pointsSorted[currentNode, 0]))
                        yNew = pointsSorted[chosen_j, 1] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,1]-pointsSorted[currentNode, 1]))
                        zNew = pointsSorted[chosen_j, 2] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,2]-pointsSorted[currentNode, 2]))
                        Z[i,0,t] = xNew.item() # x coord of the node for transition with interpolation for where the particle is if not at node
                        Z[i,1,t] = yNew.item() # y coord of the node for transition
                        Z[i,2,t] = zNew.item() # z coord of the node for transition
                        break
                    break    
                else:
                    D = alphaRW * V[currentNode, chosen_j] + Dm
                    
                    if a2 >0:
                        znTransTime = (np.sqrt(D**2*a2**4+ 2*D*distances[currentNode, chosen_j]*V[currentNode, chosen_j]*a2**2) + D*a2**2 + distances[currentNode, chosen_j]*V[currentNode, chosen_j])/V[currentNode, chosen_j]**2
    
                    else:
                        znTransTime = (- np.sqrt(D**2*a2**4+ 2*D*distances[currentNode, chosen_j]*V[currentNode, chosen_j]*a2**2) + D*a2**2 + distances[currentNode, chosen_j]*V[currentNode, chosen_j])/V[currentNode, chosen_j]**2
                    periodTransTime += znTransTime
                    velocityInterp = distances[currentNode, chosen_j]/znTransTime
                    
                    particleTotalVelocity_distWeighted = particleTotalVelocity_distWeighted + (velocityInterp*distances[currentNode, chosen_j])
                    particletotalDistance = particletotalDistance + distances[currentNode, chosen_j]
                    particleAvgVelocity_distWeighted[i] = particleTotalVelocity_distWeighted / particletotalDistance
                    
                    particleTotalVelocity = particleTotalVelocity + velocityInterp 
                    particleVelCount += 1

                #This looks to see if the particle has traveled longer than the timestep
                #if it has then the particle is moved back to where it would have been based on a linear interpolation method
                if periodTransTime >= T[t]:   
                    timeDif = periodTransTime - T[t]
                    if timeDif >= timestep:
                        while timeDif >= timestep:
                            interpPos = timeDif * distances[currentNode, chosen_j]/znTransTime

                            d = (((pointsSorted[currentNode, 0] - pointsSorted[chosen_j,0])**2)+((pointsSorted[currentNode, 1] - pointsSorted[chosen_j,1])**2)+((pointsSorted[currentNode, 2] - pointsSorted[chosen_j,2])**2))**0.5
                            xNew = pointsSorted[chosen_j, 0] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,0]-pointsSorted[currentNode, 0]))
                            yNew = pointsSorted[chosen_j, 1] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,1]-pointsSorted[currentNode, 1]))
                            zNew = pointsSorted[chosen_j, 2] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,2]-pointsSorted[currentNode, 2]))
                            
                            try:
                                Z[i,0,t] = xNew.item() # x coord of the node for transition with interpolation for where the particle is if not at node
                                Z[i,1,t] = yNew.item() # y coord of the node for transition
                                Z[i,2,t] = zNew.item() # z coord of the node for transition
                                particleTimeVelocity[i,t] = velocityInterp 
                            except IndexError:
                                t-=1
                                Z[i,0,t:] = Z[i,0,t] # x coord of the node for transition with interpolation for where the particle is if not at node
                                Z[i,1,t:] = Z[i,0,t] # y coord of the node for transition
                                Z[i,2,t:] = Z[i,0,t] # z coord of the node for transition
                                particleAvgVelocity_distWeighted[i] = particleTotalVelocity_distWeighted / particletotalDistance
                                particleTimeVelocity[i,t] = velocityInterp 
                                break
                            t+=1
                            timeDif = timeDif - timestep
    
                    try:
                        interpPos = timeDif * distances[currentNode, chosen_j]/znTransTime

                        particleTimeVelocity[i,t] = velocityInterp
                        d = (((pointsSorted[currentNode, 0] - pointsSorted[chosen_j,0])**2)+((pointsSorted[currentNode, 1] - pointsSorted[chosen_j,1])**2)+((pointsSorted[currentNode, 2] - pointsSorted[chosen_j,2])**2))**0.5
                        xNew = pointsSorted[chosen_j, 0] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,0]-pointsSorted[currentNode, 0]))
                        yNew = pointsSorted[chosen_j, 1] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,1]-pointsSorted[currentNode, 1]))
                        zNew = pointsSorted[chosen_j, 2] - ((1-((d-interpPos)/d))*(pointsSorted[chosen_j,2]-pointsSorted[currentNode, 2]))
                        Z[i,0,t] = xNew.item() # x coord of the node for transition with interpolation for where the particle is if not at node
                        Z[i,1,t] = yNew.item() # y coord of the node for transition
                        Z[i,2,t] = zNew.item() # z coord of the node for transition
                    except IndexError:
                        break
    
                currentNode = chosen_j 
            #Look to see if the particle has been recorded as making breakthrough and break the loop to go on to the next particle if so
            if Btc[i].any() != 0:
                break
            #Look to see if the particle was in the system longer than the model run time and break the loop to go on to the next particle if so
            if periodTransTime >= model_run_time:
                break
            
    toc = time.perf_counter()    
    print(f"Run time: {toc - tic:0.4f} seconds")        
    totalPV = np.sum(PV)/2
    totalPVinML = totalPV *1E6 # m^3 to mL
    
    
    
    #BTT plot from model
    BtcStack = Btc[Btc[:,:]!=0]
    BtcSorted = BtcStack[BtcStack.argsort()]
    # fraction of particles leaving core
    particles = np.arange(0,len(BtcSorted),1)
    frac_particles = particles/len(particles)
    # BtcSorted = BtcSorted[BtcSorted!=0]
    plt.plot(BtcSorted/60, frac_particles, c='k')
    # plt.legend(loc ='best', prop={'size': 8})
    plt.xlabel('Time (minutes)')
    plt.ylabel('Fraction of Particles')
    plt.title('Breakthrough Time')
    plt.show()
    
    plt.plot(BtcSorted, frac_particles, c='k')
    # plt.legend(loc ='best', prop={'size': 8})
    plt.xlabel('Time (sec)')
    plt.ylabel('Fraction of Particles')
    plt.title('Breakthrough Time')
    plt.show()
    
    tinterp = np.linspace(0, model_run_time, model_run_time)
    
    btinterp = np.interp(tinterp, BtcSorted, particles)
    btc = np.gradient(btinterp, tinterp)
    plt.plot(tinterp, btc, c='r')
    
    # double check integral is 1
    zero_t_moment = np.trapz(btc, tinterp)
    
        
    nn, hist_bins, patches = plt.hist(x=BtcSorted, bins=model_run_time, color='black',
                                alpha=0.7, rwidth=1, density=False)

    xcenters = (hist_bins[:-1] + hist_bins[1:]) / 2
    width = 1 * (hist_bins[1] - hist_bins[0])
    histNorm = nn/(np.sum(nn))
    
    #3d plot through time to track particles
    #####uncomment this if you want to see the 3d plot
    # for i in range(10):
    #     fig = plt.figure(figsize=(18,6),dpi=300)
    #     # ax = Axes3D(fig)
    #     ax = fig.add_subplot(projection='3d')
        
    #     # for j in range(len(inlet_ind)):
    #     # inlet point
    #     scI = ax.scatter(pointsSorted[inlet_ind,0], pointsSorted[inlet_ind,1],pointsSorted[inlet_ind,2], c='green', s=100, alpha = 1)
    #     # plot outlet point
    #     scO = ax.scatter( pointsSorted[outlet_ind,0], pointsSorted[outlet_ind,1],pointsSorted[outlet_ind,2], c='orange', s=100, alpha = 1)
    #     # Plot points based on aperature size
    #     amax = np.max(aperature)
    #     sc = ax.scatter(pointsSorted[:,0], pointsSorted[:,1], pointsSorted[:,2], c='black', s=aperature/amax*100)
    
    #     sc2 = ax.scatter(Z[:,0,i], Z[:,1,i], Z[:,2,i], c='red', s=3)

    #     plt.xticks(fontsize=22)
    #     plt.yticks(fontsize=22)
    #     ax.zaxis.set_tick_params(labelsize=22)
        
    #     plt.xlim([0,0.04])
    #     plt.ylim([0,0.05])
    #     ax.set_zlim(0,0.1)
    #     ax.set_aspect('equal')
    #     plt.xlabel('meters')
    #     plt.ylabel('meters')
    #     plt.tight_layout()
    #     plt.show()
    #########end of 3d plot
    
    particle_rad = flowrate * np.trapz(granPoint5_rad2,trad2) / zn
    
    C_out = np.zeros((len(nn),1))
    C_outNorm = np.zeros((len(nn),1))
    for i in range(len(nn)):# C_out[i] = (particle_radCalc(flowrate,granPoint5_rad2,trad2,len(BtcSorted),hist_bins[i]) * (nn[i]/(flowrate*width))) # - (np.trapz(granPoint5_rad2,trad2) * np.exp(-(np.log(2))*hist_bins[-1]/(109.7*60)) / (zn *nn[-1]))
        C_out[i] = particle_rad * nn[i] / (flowrate*width)
        
    C_outNorm = C_out/np.max(granPoint5_rad2)
    
    quant_exp = quantile_calc(granpoint5_rad3Norm, trad3, 0.5)
    quant_model = quantile_calc(histNorm, xcenters, 0.5)
    quant_model2 = quantile_calc(btc, tinterp, 0.5)
    
    
    particle_rad = flowrate * np.trapz(granPoint5_rad2,trad2) / zn
    
    C_out2 = np.zeros((len(btc),1))
    C_outNorm2 = np.zeros((len(btc),1))
    for i in range(len(btc)):
        C_out2[i] = particle_rad * btc[i] / (flowrate*(tinterp[1]-tinterp[0]))
        
    C_outNorm2 = C_out2/np.max(granPoint5_rad2)

    return (V, btc, tinterp, quant_exp, quant_model2, C_outNorm2[:,0],totalPV,particleAvgVelocity[:,0],particleAvgVelocity_distWeighted[:,0])


def modelPlots(trad3, granpoint5_rad3Norm, tinterp, quant_exp, quant_model2, C_outNorm2):

    plt.figure(figsize=(14,6))
    # plt.plot((trad3/60)*0.5/3, granpoint5_rad3Norm, color = "blue", label = '$^{18}$F[FDG] Tracer') # experimental data
    # plt.plot(tinterp*flowrate/(totalPV+(PVofInjection)), C_outNorm2[:,0], color = "black", label = 'Particle Random Walk Model Data', linewidth = 0.5) # graph model
    plt.plot(trad3, granpoint5_rad3Norm, color = "blue", label = '$^{18}$F[FDG] Tracer') # experimental data
    plt.plot(tinterp+((quant_exp-quant_model2)), C_outNorm2, color = "black", label = 'Particle Random Walk Model', linewidth = 0.5) # graph model
    
    # Format the plots
    plt.legend(fontsize=22)
    # plt.xlim([0,0.5])
    plt.ylim([0,0.5])
    
    # plt.yscale('log')
    plt.xlabel('Time (model norm based on experimental dead volume)', fontsize = 28)
    plt.ylabel('C/C$_0$', fontsize = 28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.show()
    
    plt.figure(figsize=(14,6))
    # plt.plot((trad3/60)*0.5/3, granpoint5_rad3Norm, color = "blue", label = '$^{18}$F[FDG] Tracer') # experimental data
    # plt.plot(tinterp*flowrate/(totalPV+(PVofInjection)), C_outNorm2[:,0], color = "black", label = 'Particle Random Walk Model Data', linewidth = 0.5) # graph model
    plt.plot(trad3, granpoint5_rad3Norm, color = "blue", label = '$^{18}$F[FDG] Tracer') # experimental data
    plt.plot(tinterp+((quant_exp-quant_model2)), C_outNorm2, color = "black", label = 'Particle Random Walk Model', linewidth = 0.5) # graph model
    
    # Format the plots
    plt.legend(fontsize=22)
    # plt.xlim([0,0.5])
    plt.ylim([9E-3,0.5])
    
    plt.yscale('log')
    plt.xlabel('Time (model norm based on experimental dead volume)', fontsize = 28)
    plt.ylabel('C/C$_0$', fontsize = 28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.show()