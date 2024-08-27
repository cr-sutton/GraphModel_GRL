# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:58:07 2024

@author: Collin Sutton
prepared for "A laboratory-validated, graph-based flow and transport model for naturally fractured media" 
submitted to Geophysical Research Letters
Reuse of this code must cite the original paper
"""

# Load packages
# Only packages called in this script need to be imported
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate
from sklearn.metrics.pairwise import euclidean_distances

plt.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 18})

# load data
# tracer filenames
##replace this with the path to the .raw file that you download
filename = 'C:\\\\Users\\colli\\Documents\\Research\\GraniteFracture\\PET\\21Nov2022\\21Nov2022_granite_point5mL_60s_64_64_159_20.raw'

# timestep size depending on reconstruction. This information can also be found in the header files
timestep_size = 1 # min
flow_rate = 0.5 # ml/min
injection_loop_delay = 2 #min
# manually set the dimensions of the raw file
img_dim = [64, 64, 159, 20]

# crop off ends
z_axis_crop = [35, 148] # granite 0.5ml/min

# voxel size. This information can be found in the .hdr img file. 
vox_size = [0.07763, 0.07763, 0.0796] # (These are the default values)
t = np.arange((timestep_size/2), (img_dim[3]-1)*timestep_size, timestep_size)

# Define a function for loading the PET data
def load_reshape_pet(filename, img_dim, z_axis_crop):
    ## LOAD DATA and reshape
    # raw_data = np.fromfile((path2data + '\\' + filename), dtype=np.float32)
    raw_data = np.fromfile((filename), dtype=np.float32)
    raw_data = np.reshape(raw_data, (img_dim[3], img_dim[2], img_dim[1], img_dim[0]))
    raw_data = np.transpose(raw_data, (2, 3, 1, 0))
    # flip so that correctly oriented on slice plots with 0,0 in lower left
    raw_data = np.flip(raw_data, axis=0)
    # crop extra long timesteps
    raw_data = raw_data[:,:,z_axis_crop[0]:z_axis_crop[1],1:]
    # flip sp that tracer flows from left to right
    # raw_data = np.flip(raw_data, axis=2)
    return raw_data

# Define 3D grid corresponding to image voxels
def grid_gen(mat3d, vox_size):
    r, c, s = np.shape(mat3d)
    x_coord = np.linspace(vox_size[0]/2, vox_size[0]*c, c+1)
    y_coord = np.linspace(vox_size[1]/2, vox_size[1]*r, r+1)
    s_coord = np.linspace(vox_size[2]/2, vox_size[2]*s, s+1)
    X, Y, Z = np.meshgrid(x_coord, y_coord, s_coord, indexing='xy')
    return X,Y,Z

def plot_2d(map_data, dx, dy, colorbar_label, cmap, *args):
    # plt.figure(figsize=(18,6),dpi=300)
    r, c = np.shape(map_data)
    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    X, Y = np.meshgrid(x_coord, y_coord)
    # Note that we are flipping map_data and the yaxis to so that y increases downward
    plt.figure(figsize=(14, 3), dpi=300)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'auto', edgecolor ='none', linewidth = 0.01)
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    if args:
        plt.clim(cmin, cmax) 
    # label the colorbar
    cbar.set_label(colorbar_label)
    # make axis fontsize bigger!
    # plt.tick_params(axis='both', which='major')
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 

def scatterplot3d(Xs, Zs, Ys, Zms, cmap='magma_r'): #make a new one with option for color scale and for colorbar label
    # fig = plt.figure(dpi=300)
    fig = plt.figure(figsize=(18,6),dpi=300)
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(Xs,  Zs, Ys, s = 5, c= Zms, cmap=cmap, edgecolors='none')
    plt.colorbar(sc, label="[-]")
    ax.set_xlabel('Inlet face')
    ax.set_ylabel('\nDistance from inlet [cm]')
    ax.set_aspect('equal')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
def scatterplot3d2(Xs, Zs, Ys, Zms,vmin,vmax, cmap='magma_r'):
    # fig = plt.figure(dpi=300)
    fig = plt.figure(figsize=(12,6),dpi=300)
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(Xs,  Zs, Ys, s = 5, c= Zms, cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.2, edgecolors='none')
    # cbar = plt.colorbar(sc,label = "(-)",orientation='horizontal' )
    # cbar.ax.tick_params(labelsize=28)
    # cbar.ax.set_title('(-)',fontsize=28)
    # ax.set_xlabel('Inlet face (cm)')
    # ax.set_ylabel('\nDistance from inlet (cm)')
    # # plt.zlabel("Position along core axis")
    # ax.set_zlabel('\nPosition along core axis (cm)')
    ax.set_aspect('equal')
    ax.view_init(-140, 60)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.tick_params(labelsize=16)
    # ax.set_zticks([])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    # for axis in [ax.zaxis]:
        axis.set_ticklabels([])
        axis._axinfo['axisline']['linewidth'] = 1
        axis._axinfo['axisline']['color'] = (0, 0, 0)
        axis._axinfo['grid']['linewidth'] = 0.5
        axis._axinfo['grid']['linestyle'] = "-"
        axis._axinfo['grid']['color'] = (0, 0, 0)
        axis._axinfo['tick']['inward_factor'] = 0.0
        axis._axinfo['tick']['outward_factor'] = 0.0
        # axis.set_pane_color((0.95, 0.95, 0.95))

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.zaxis.set_tick_params(labelsize=22)
    
# zero temporal moment
def zero_moment(raw_data, timearray):
    array_size = raw_data.shape
    Zm = np.zeros(array_size[:-1])
    Marriv = np.zeros(array_size[:-1])
    for i in range(array_size[0]):
        for j in range(array_size[1]):
            for k in range(array_size[2]):
                zeromoment = np.trapz(raw_data[i,j,k,:], timearray)
                Zm[i,j,k] = zeromoment
                firstmoment = np.trapz(timearray*raw_data[i,j,k,:], timearray)
                Marriv[i,j,k] = firstmoment/zeromoment
            
    Zm = Zm/np.max(Zm)
    return Zm, Marriv

    
# Much faster quantile calculation 
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


# Function to calculate the quantile arrival time map
def exp_arrival_map_function(conc, timestep, grid_size, quantile):
    # start timer
    tic = time.perf_counter()
    # determine the size of the data
    conc_size = conc.shape
    # define array of times based on timestep size (in seconds)
    # Note that we are referencing the center of the imaging timestep since a 
    # given frame is an average of the detected emission events during that period
    timearray = np.arange(timestep/2, timestep*conc_size[3], timestep)
    
    # sum of slice concentrations for calculating inlet and outlet breakthrough
    oned = np.nansum(np.nansum(conc, 0), 0)
    
    # arrival time calculation in inlet slice
    tau_in = quantile_calc(oned[0,:], timearray, quantile)
    
    # arrival time calculation in outlet slice
    tau_out = quantile_calc(oned[-1,:], timearray, quantile)

    # core length
    core_length = grid_size[2]*conc_size[2]
    # array of grid cell centers before interpolation
    z_coord_pet = np.arange(grid_size[2]/2, core_length, grid_size[2])
    
    # Preallocate arrival time array
    at_array = np.zeros((conc_size[0], conc_size[1], conc_size[2]), dtype=float)
    
    for xc in range(0, conc_size[0]):
        for yc in range(0, conc_size[1]):
            for zc in range(0, conc_size[2]):
                # Check if outside core
                if np.isfinite(conc[xc, yc, zc, 0]):
                    # extract voxel breakthrough curve
                    cell_btc = conc[xc, yc, zc, :]
                    # check to make sure tracer is in grid cell
                    if cell_btc.sum() > 0:
                        # call function to find quantile of interest
                        at_array[xc, yc, zc] = quantile_calc(cell_btc, timearray, quantile)

    v = (core_length-grid_size[2])/(tau_out - tau_in)
    print('advection velocity: ' + str(v))
    
    # Normalize arrival times
    at_array_norm = (at_array-tau_in)/(tau_out - tau_in)
    
    # vector of ideal mean arrival time based average v
    at_ideal = z_coord_pet/z_coord_pet[-1]

    # Turn this vector into a matrix so that it can simply be subtracted from
    at_ideal_array = np.tile(at_ideal, (conc_size[0], conc_size[1], 1))

    # Arrival time difference map
    at_diff_norm = (at_ideal_array - at_array_norm)
    
    # Replace nans with zeros
    at_array[np.isnan(conc[:,:,:,0])] = 0
    # Replace nans with zeros
    at_array_norm[np.isnan(conc[:,:,:,0])] = 0
    # Replace nans with zeros
    at_diff_norm[np.isnan(conc[:,:,:,0])] = 0
    # stop timer
    toc = time.perf_counter()
    print(f"Function runtime is {toc - tic:0.4f} seconds")
    return at_array, at_array_norm, at_diff_norm

# call function to load tracer data
raw_data = load_reshape_pet(filename, img_dim, z_axis_crop)
# raw_data = np.flip(raw_data, axis=2 )# flip left <--> right. Tracer now enters from the left
raw_data = raw_data[2:62,2:62,:,:]
raw_data = raw_data/.5 #normalize by max 

# for t in range(img_dim[-1]):
cmax_raw_data = np.zeros_like(raw_data[:,:,:,1])
    
for i in range(len(raw_data[:,0,0,0])):
    for j in range(len(raw_data[0,:,0,0])):
        for k in range(len(raw_data[0,0,:,0])):
            cmax_raw_data[i,j,k] = np.max(raw_data[i,j,k,:])

X, Y, Z = grid_gen(cmax_raw_data, vox_size)
# Define arrays of thresholded locations and zero moments
# Cm_thresh = (cmax_raw_data>0.1).nonzero()
Cm_thresh = (cmax_raw_data>0.05).nonzero()
Xs = X[Cm_thresh]
Ys = Y[Cm_thresh]
Zs = Z[Cm_thresh]
Zms = cmax_raw_data[Cm_thresh]
scatterplot3d2(Xs, Zs, Ys, cmax_raw_data[Cm_thresh],0.05,.4, cmap = "Reds")
scatterplot3d2(Xs,  Zs, Ys, cmax_raw_data[Cm_thresh],0.0,0.45,cmap = "Oranges")

graphData_cm = np.column_stack((Xs,Ys,Zs,cmax_raw_data[Cm_thresh]))

# np.savetxt('PETGraph_cMax_point1'+'.txt', graphData_cm)

# timestep
ts1 = 7
# Call plot 2D function
plot_2d(np.sum(raw_data[:,:,:,ts1], axis=0), vox_size[0], vox_size[2], '[-]', cmap='Reds')
plt.clim(0.05, 1)
vi = ((ts1*timestep_size) - injection_loop_delay )*flow_rate
# plt.title('Tracer, VI=%1.2f mL' %vi)
plt.xlabel("Position along core axis [cm]", fontsize = 16)
plt.ylabel("Core face [cm]", fontsize = 16)
plt.show()

Zm, Fm = zero_moment(raw_data, t)
at_array, at_array_norm, at_diff_norm = exp_arrival_map_function(raw_data, timestep_size, vox_size, 0.5)

# plot zero moment
slice = 28
plot_2d(np.squeeze(Zm[slice,:,:]), vox_size[0], vox_size[2], '[-]', cmap='magma_r')
plt.xlabel("Position along core axis [cm]", fontsize = 20)
plt.ylabel("Core face [cm]", fontsize = 20)
plt.clim([0.05, 0.3])
plt.show()


# get the indexing of the voxesl in Zm greather than the threshold
zm_threshold = 0.1
Zm_thresh = (Zm>zm_threshold).nonzero() 

X, Y, Z = grid_gen(Zm, vox_size)
# Define arrays of thresholded locations and zero moments
Xs = X[Zm_thresh]
Ys = Y[Zm_thresh]
Zs = Z[Zm_thresh]
Zms = Zm[Zm_thresh]
for i in range(11):
    ts1 = i
    c1 = raw_data[:,:,:,ts1]
    scatterplot3d2(Xs, Zs, Ys, c1[Zm_thresh],0.0,0.3, cmap = "Reds")


#%%
###This makes the plot in Figure 2. 
###Comment or uncomment depending on which graph size you want
###.npy files are generated in the particle tracking code

# n = np.load('n_15.npy')
# Q= np.load('Q_15.npy')
# pointsSorted=np.load('pointsSorted_15.npy')
# aperature=np.load('aperature_15.npy')
# inlet_ind=np.load('inlet_ind_15.npy')
# outlet_ind=np.load('outlet_ind_15.npy')
n = np.load('n_10.npy')
Q= np.load('Q_10.npy')
pointsSorted=np.load('pointsSorted_10.npy')
aperature=np.load('aperature_10.npy')
inlet_ind=np.load('inlet_ind_10.npy')
outlet_ind=np.load('outlet_ind_10.npy')
# n = np.load('n_17.npy')
# Q= np.load('Q_17.npy')
# pointsSorted=np.load('pointsSorted_17.npy')
# aperature=np.load('aperature_17.npy')
# inlet_ind=np.load('inlet_ind_17.npy')
# outlet_ind=np.load('outlet_ind_17.npy')

pointsSorted=pointsSorted*100

# plot the graph
fig = plt.figure(figsize=(12,6),dpi=300)
#fig = plt.figure(figsize=(18,6),dpi=300)
ax = fig.add_subplot(projection='3d')

# c1 = raw_data[:,:,:,4]
# sc=ax.scatter(Xs,  Zs, Ys, s = 5, c= c1[Zm_thresh], alpha = 0.3,cmap = "Reds",vmin=0,vmax=0.3, edgecolors='none')



X, Y, Z = grid_gen(cmax_raw_data, vox_size)
# Define arrays of thresholded locations and zero moments
# Cm_thresh = (cmax_raw_data>0.1).nonzero()
Cm_thresh = (cmax_raw_data>0.05).nonzero()
Xs = X[Cm_thresh]
Ys = Y[Cm_thresh]
Zs = Z[Cm_thresh]

sc=ax.scatter(Xs,  Zs, Ys, c=cmax_raw_data[Cm_thresh],vmin=0.0,vmax=0.45, alpha = 0.05,cmap = "Oranges", edgecolors='none')

# plt.savefig('fracturePlaneInjection' + str(i) + '.png')

# plot connections
qmax = np.max(Q)
# loop through rows of the Q matrxi
for i in range(n):
    for j in range(i, n, 1):
        if abs(Q[i,j]) > 0:
            # plt.plot([pointsSorted[i,2], pointsSorted[j,2]], [pointsSorted[i,0], pointsSorted[j,0]], color='k', linewidth = 3*abs(Q[i,j])/qmax)
            # ax.plot([pointsSorted[i,2], pointsSorted[j,2]], [pointsSorted[i,0], pointsSorted[j,0]], color='k', linewidth = 3*abs(Q[i,j])/qmax)
            ax.plot([pointsSorted[i,0], pointsSorted[j,0]], [pointsSorted[i,2], pointsSorted[j,2]], [pointsSorted[i,1], pointsSorted[j,1]],color='k', linewidth = 3*abs(Q[i,j])/qmax)


ax.scatter(pointsSorted[:,0], pointsSorted[:,2], pointsSorted[:,1], c='black', s=30)
# Plot points based on aperature size
amax = np.max(aperature)

# inlet point

ax.plot(pointsSorted[inlet_ind,0], pointsSorted[inlet_ind,2],pointsSorted[inlet_ind,1], 'og')

# plot outlet point

ax.plot( pointsSorted[outlet_ind,0], pointsSorted[outlet_ind,2], pointsSorted[outlet_ind,1],'or')


# cbar = plt.colorbar(sc)
# cbar.solids.set(alpha=1)
# ax.tick_params(labelsize=22)
# ax.set_xlabel('Inlet face',fontsize = 22)
# ax.set_ylabel('\nDistance from inlet (cm)',fontsize = 22)
ax.set_aspect('equal')
ax.view_init(-140, 60)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
ax.zaxis.set_tick_params(labelsize=0)
# for axis in [ax.xaxis, ax.yaxis,ax.zaxis]:
for axis in [ax.zaxis]:
    axis.set_ticklabels([])
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = (0, 0, 0)
    axis._axinfo['grid']['linewidth'] = 0.5
    axis._axinfo['grid']['linestyle'] = "-"
    axis._axinfo['grid']['color'] = (0, 0, 0)
    axis._axinfo['tick']['inward_factor'] = 0.0
    axis._axinfo['tick']['outward_factor'] = 0.0
plt.tight_layout()
plt.show()

