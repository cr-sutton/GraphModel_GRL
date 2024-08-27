# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:58:07 2024

@author: Collin Sutton
prepared for "A laboratory-validated, graph-based flow and transport model for naturally fractured media" 
submitted to Geophysical Research Letters
Reuse of this code must cite the original paper
"""
import numpy as np
from scipy.stats import norm 
import scipy.stats as ss
import matplotlib.pyplot as plt
import os
import scipy

# os.chdir("c:\\\\Users\\colli\\Documents\\Python Scripts")
os.chdir("c:\\\\Users\\colli\\Desktop\\modelRunFiles")
plt.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 18})

from graphFunctions_andPT_GRL import quantile_calc,inputExpData,graphModel,modelPlots

#%%
######This is used to many models for different flow rates for one graph
graph_filename = "granite_frac_graphFromPET_cMax_HDBSCAN_minClusterSize_15"
filename = ['C:\\\\Users\\colli\\Documents\\Python Scripts\\granite_1mL_rad2_3.csv','C:\\\\Users\\colli\\Documents\\Python Scripts\\granite_1ml_200ulLoop_rad2_3.csv', \
             'C:\\\\Users\\colli\\Documents\\Python Scripts\\granite_point5_rad2_3.csv', 'C:\\\\Users\\colli\\Documents\\Python Scripts\\granite_point5_200ulLoop_rad2_3.csv']

looptime = [94,50,190,90]
flowrate = [1.66667E-8,1.66667E-8,8.34E-9,8.34E-9] #[m^3/s] - 1 mL/min, 0.5 mL/min
model_run_time = [60*10,60*11,60*20,60*20]
alphaRW = [0.165,0.165,0.165,0.165] # this is for 15 graph

q = [1,1,0.5,0.5]
tubingVolume = [2.6,2.6,1.8,1.8]
loopSize = [1,1,0.2,0.2]
firstArrival = [97,97,196,196]

deadVolumeTime = np.zeros((len(filename)))

trad3 = {}
granpoint5_rad3Norm = {}
C_outNorm2 = {}
tinterp = {}
C_outNorm2[0] = np.zeros((model_run_time[0]))
tinterp[0] = np.zeros((model_run_time[0]))
C_outNorm2[1] = np.zeros((model_run_time[1]))
tinterp[1] = np.zeros((model_run_time[1]))
C_outNorm2[2] = np.zeros((model_run_time[2]))
tinterp[2] = np.zeros((model_run_time[2]))
C_outNorm2[3] = np.zeros((model_run_time[3]))
tinterp[3] = np.zeros((model_run_time[3]))

quantiles = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
quantileCalcsExp = np.zeros((len(filename),len(quantiles)))
quantileCalcsModel = np.zeros((len(filename),len(quantiles)))
#####model plots in the manuscript use 1,000,000 particles while this code is set to 100,000 for speed
#####change the below 2 parameters and the zn in the "graphFunctions_andPT_GRL.py" script to change number of particles ran
particleAvgVelocity = np.zeros((100000,len(filename)))
particleAvgVelocity_distWeighted = np.zeros((100000,len(filename)))

for i in range(len(filename)):
    trad2, trad3[i],granPoint5_rad2,granPoint5_rad3,granpoint5_rad2RTD,granpoint5_rad3RTD,granpoint5_rad2Norm,granpoint5_rad3Norm[i] = inputExpData(filename[i], looptime[i])
    dy = 0.003 #[m]
    alpha = 0.7 #[-]

    V, btc, tinterp[i], quant_exp, quant_model2, C_outNorm2[i], pv, particleAvgVelocity[:,i],particleAvgVelocity_distWeighted[:,i] = graphModel(trad2, trad3[i],granPoint5_rad2, granPoint5_rad3, granpoint5_rad2RTD, granpoint5_rad3RTD, granpoint5_rad2Norm, granpoint5_rad3Norm[i], graph_filename, dy, alpha, flowrate[i], model_run_time[i], alphaRW[i])

    deadVolumeTime[i] = (tubingVolume[i]-loopSize[i])/q[i]*60
    modelPlots(trad3[i], granpoint5_rad3Norm[i], tinterp[i], deadVolumeTime[i], 0, C_outNorm2[i])


    
    for j in range(len(quantiles)):
        quantileCalcsExp[i,j] = quantile_calc(granpoint5_rad3Norm[i], trad3[i], quantiles[j])
        quantileCalcsModel[i,j] = quantile_calc(C_outNorm2[i], tinterp[i]+deadVolumeTime[i], quantiles[j])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(quantileCalcsModel[i], quantileCalcsExp[i], s=10, c='black', marker="s", label='Experimental Quantiles')
    plt.title("Flowrate = %f" % q[i])
    plt.xlabel("Quantile")
    plt.ylabel("Time [s]")
    plt.show()
    



plt.figure(figsize=(12,6), dpi = 300)
cmap = plt.get_cmap('Reds')
cmapModel = plt.get_cmap('Greys')
labels = ['1 mL/min - 1 mL Pulse', '1 mL/min - 200 $\mu$L Pulse','0.5 mL/min - 1 mL Pulse','0.5 mL/min - 200 $\mu$L Pulse',]
for i in range(len(filename)):
    plt.plot(trad3[i], granpoint5_rad3Norm[i], color = cmap((i+1)/len(filename)), linestyle='--', label = labels[i], linewidth = 2) # experimental data
    plt.plot(tinterp[i]+deadVolumeTime[i], C_outNorm2[i], color = cmapModel((i+1)/len(filename)), linewidth = 2) # graph model
plt.xlim([0,1000])
plt.ylim([9E-3,0.5])
plt.xlabel('Time (s)', fontsize = 28)
plt.ylabel('C/C$_0$ (-)', fontsize = 28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()

plt.figure(figsize=(12,6), dpi = 300)
cmap = plt.get_cmap('Reds')
cmapModel = plt.get_cmap('Greys')
labels = ['1 mL/min - 1 mL Pulse', '1 mL/min - 200 $/mu$L Pulse','0.5 mL/min - 1 mL Pulse','0.5 mL/min - 200 $/mu$L Pulse',]
for i in range(len(filename)):
    plt.plot(trad3[i], granpoint5_rad3Norm[i], color = cmap((i+1)/len(filename)), linestyle='--', label = labels[i], linewidth = 2) # experimental data
    plt.plot(tinterp[i]+deadVolumeTime[i], C_outNorm2[i], color = cmapModel((i+1)/len(filename)), linewidth = 2) # graph model
plt.xlim([0,1000])
plt.ylim([9E-3,0.5])

plt.yscale('log')
plt.xlabel('Time (s)', fontsize = 28)
plt.ylabel('C/C$_0$ (-)', fontsize = 28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()


###plt each flow rate separetly
plt.figure(figsize=(8,6), dpi = 300)
cmap = plt.get_cmap('Reds')
cmapModel = plt.get_cmap('Greys')
labels = ['1 mL Pulse', '200 $\mu$L Pulse']
plt.plot(trad3[0], granpoint5_rad3Norm[0], color = cmap((2+1)/len(filename)), linestyle='--', label = labels[0], linewidth = 2) # experimental data
plt.plot(trad3[1], granpoint5_rad3Norm[1], color = cmap((3+1)/len(filename)), linestyle='--', label = labels[1], linewidth = 2) # experimental data
plt.plot(tinterp[0]+deadVolumeTime[0], C_outNorm2[0], color = cmapModel((2+1)/len(filename)), linewidth = 2) # graph model
plt.plot(tinterp[1]+deadVolumeTime[1], C_outNorm2[1], color = cmapModel((3+1)/len(filename)), linewidth = 2) # graph model
plt.xlim([0,500])
plt.ylim([9E-3,0.5])
plt.xlabel('Time (s)', fontsize = 32)
plt.ylabel('C/C$_0$ (-)', fontsize = 32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()

plt.figure(figsize=(8,6), dpi = 300)
cmap = plt.get_cmap('Reds')
cmapModel = plt.get_cmap('Greys')
labels = ['1 mL Pulse', '200 $\mu$L Pulse']
plt.plot(trad3[0], granpoint5_rad3Norm[0], color = cmap((2+1)/len(filename)), linestyle='--', label = labels[0], linewidth = 2) # experimental data
plt.plot(trad3[1], granpoint5_rad3Norm[1], color = cmap((3+1)/len(filename)), linestyle='--', label = labels[1], linewidth = 2) # experimental data
plt.plot(tinterp[0]+deadVolumeTime[0], C_outNorm2[0], color = cmapModel((2+1)/len(filename)), linewidth = 2) # graph model
plt.plot(tinterp[1]+deadVolumeTime[1], C_outNorm2[1], color = cmapModel((3+1)/len(filename)), linewidth = 2) # graph model
plt.xlim([0,500])
plt.ylim([9E-3,0.5])

plt.yscale('log')
plt.xlabel('Time (s)', fontsize = 32)
plt.ylabel('C/C$_0$ (-)', fontsize = 32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()

plt.figure(figsize=(8,6), dpi = 300)
cmap = plt.get_cmap('Reds')
cmapModel = plt.get_cmap('Greys')
labels = ['1 mL Pulse', '200 $\mu$L Pulse']
plt.plot(trad3[2], granpoint5_rad3Norm[2], color = cmap((2+1)/len(filename)), linestyle='--', label = labels[0], linewidth = 2) # experimental data
plt.plot(trad3[3], granpoint5_rad3Norm[3], color = cmap((3+1)/len(filename)), linestyle='--', label = labels[1], linewidth = 2) # experimental data
plt.plot(tinterp[2]+deadVolumeTime[2], C_outNorm2[2], color = cmapModel((2+1)/len(filename)), linewidth = 2) # graph model
plt.plot(tinterp[3]+deadVolumeTime[3], C_outNorm2[3], color = cmapModel((3+1)/len(filename)), linewidth = 2) # graph model
plt.xlim([0,1000])
plt.ylim([9E-3,0.5])
plt.xlabel('Time (s)', fontsize = 32)
plt.ylabel('C/C$_0$ (-)', fontsize = 32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()

plt.figure(figsize=(8,6), dpi = 300)
cmap = plt.get_cmap('Reds')
cmapModel = plt.get_cmap('Greys')
labels = ['1 mL Pulse', '200 $\mu$L Pulse']
plt.plot(trad3[2], granpoint5_rad3Norm[2], color = cmap((2+1)/len(filename)), linestyle='--', label = labels[0], linewidth = 2) # experimental data
plt.plot(trad3[3], granpoint5_rad3Norm[3], color = cmap((3+1)/len(filename)), linestyle='--', label = labels[1], linewidth = 2) # experimental data
plt.plot(tinterp[2]+deadVolumeTime[2], C_outNorm2[2], color = cmapModel((2+1)/len(filename)), linewidth = 2) # graph model
plt.plot(tinterp[3]+deadVolumeTime[3], C_outNorm2[3], color = cmapModel((3+1)/len(filename)), linewidth = 2) # graph model
plt.xlim([0,1000])
plt.ylim([9E-3,0.5])

plt.yscale('log')
plt.xlabel('Time (s)', fontsize = 32)
plt.ylabel('C/C$_0$ (-)', fontsize = 32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()




#%%
######This is used to calculate velocity distributions over a range of graph sizes for one flow rate
graph_filename = ["granite_frac_graphFromPET_cMax_HDBSCAN_minClusterSize_10",\
                  "granite_frac_graphFromPET_cMax_HDBSCAN_minClusterSize_15",\
                  "granite_frac_graphFromPET_cMax_HDBSCAN_minClusterSize_17"]
filename = 'C:\\\\Users\\colli\\Documents\\Python Scripts\\granite_point5_rad2_3.csv'

looptime = [190,190,190]
flowrate = [8.34E-9,8.34E-9,8.34E-9] #[m^3/s] - 1 mL/min, 0.5 mL/min
model_run_time = [60*20,60*20,60*20]
alphaRW = [0.06,0.165,0.45] # this is for 15 graph
q = [0.5,0.5,0.5]
tubingVolume = [2.6,2.6,2.6]
loopSize = [1,1,1]
C_outNorm2 = np.zeros((model_run_time[0],len(graph_filename)))
tinterp = np.zeros((model_run_time[0],len(graph_filename)))
deadVolumeTime = np.zeros((len(graph_filename)))
quantiles = [.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]
quantileCalcsExp = np.zeros((len(graph_filename),len(quantiles)))
quantileCalcsModel = np.zeros((len(graph_filename),len(quantiles)))
particleAvgVelocity = np.zeros((100000,len(graph_filename)))
particleAvgVelocity_distWeighted = np.zeros((100000,len(graph_filename)))
particleTimeVelocity = np.zeros((100000,21, len(graph_filename))) #particles, len(T) model_run_time / timestep + 1 , number of graphs

###this is for calculating quantiles without noise at end sliced at 0.009
C_outNorm2_sliced = {}
cOut_times = [573,581,567]

for i in range(len(graph_filename)):
    trad2, trad3,granPoint5_rad2,granPoint5_rad3,granpoint5_rad2RTD,granpoint5_rad3RTD,granpoint5_rad2Norm,granpoint5_rad3Norm = inputExpData(filename, looptime[i])
    
    dy = 0.003 #[m]
    alpha = 0.7 #[-]
    V, btc, tinterp[:,i], quant_exp, quant_model2, C_outNorm2[:,i], pv, particleAvgVelocity[:,i],particleAvgVelocity_distWeighted[:,i] = graphModel(trad2, trad3,granPoint5_rad2, granPoint5_rad3, granpoint5_rad2RTD, granpoint5_rad3RTD, granpoint5_rad2Norm, granpoint5_rad3Norm, graph_filename[i], dy, alpha, flowrate[i], model_run_time[i], alphaRW[i])

    deadVolumeTime[i] = (tubingVolume[i]-loopSize[i])/q[i]*60
    modelPlots(trad3, granpoint5_rad3Norm, tinterp[:,i], deadVolumeTime[i], 0, C_outNorm2[:,i])

    granpoint5_rad3NormSliced = granpoint5_rad3Norm[:908]
    C_outNorm2_sliced[i] = C_outNorm2[:cOut_times[i],i]
    tinterp1 = np.linspace(0, len(C_outNorm2_sliced[i]), len(C_outNorm2_sliced[i]))
    for j in range(len(quantiles)):

        quantileCalcsExp[i,j] = quantile_calc(granpoint5_rad3NormSliced, trad3[:(len(granpoint5_rad3NormSliced))], quantiles[j])
        quantileCalcsModel[i,j] = quantile_calc(C_outNorm2_sliced[i], tinterp1+deadVolumeTime[i], quantiles[j])
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(quantileCalcsModel[i], quantileCalcsExp[i], s=10, c='black', marker="s", label='Experimental Quantiles')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(quantileCalcsExp[i,:], quantileCalcsModel[i,:])
    plt.text(290, 575, 'r$^2$ = %f' % r_value**2, fontsize=16, bbox=dict(facecolor='red', alpha=0.5))
    plt.title("Flowrate = %f" % q[i])
    plt.xlabel("Quantile")
    plt.ylabel("Time [s]")
    plt.show()
    

#calculate PDF of velocity distributions
mean = np.zeros((len(graph_filename),1))
bins = 150
n = np.zeros((bins,len(graph_filename)))
x = np.zeros((bins+1,len(graph_filename)))
xcenters = np.zeros((bins,len(graph_filename)))
labels = ['G = (44,115)', 'G = (16,36)', 'G = (8,16)']
for i in range(len(graph_filename)):
    mean[i] = np.mean(particleAvgVelocity_distWeighted[:,i]) 
    particleAvgVelocity_distWeighted[:,i] =  particleAvgVelocity_distWeighted[:,i]/mean[i]
    n[:,i], x[:,i], _i = plt.hist(x=np.log10(particleAvgVelocity_distWeighted[:,i]), bins=bins, alpha=0.3, rwidth=1, density = True, label=labels[i])
    hist_bins = x[:,i]
    xcenters[:,i] = (hist_bins[:-1] + hist_bins[1:]) / 2
plt.legend(loc='best')
plt.show()

cmap = plt.get_cmap('viridis')
plt.figure(figsize=(10,6))
plt.plot(10**xcenters[:,0],n[:,0],label=labels[0], color = 'blue', linewidth = 2)
plt.plot(10**xcenters[:,1],n[:,1],label=labels[1],color = 'green',  linewidth = 2)
plt.plot(10**xcenters[:,2],n[:,2],label=labels[2],color = 'orange', linewidth = 2)
plt.xlabel(r'V/$\bar{V}$',fontsize=28)
plt.ylabel("Probability",fontsize=28)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()


#10 to the power of x1 and n1 and that should get it back to linear then plot again
plt.plot(10**x[:-1,0],n[:,0],label='Most Nodes')
plt.plot(10**x[:-1,1],n[:,1],label='Middle')
plt.plot(10**x[:-1,2],n[:,2],label='Least Nodes')
plt.legend(loc="best")
plt.xscale("log")
plt.yscale("log")
plt.show()

###btc plots for different sized graphs

cmap = plt.get_cmap('Reds')

plt.figure(figsize=(10,6), dpi = 300)
plt.plot(trad3, granpoint5_rad3Norm, color = cmap(0.9005767012687428), alpha = 1, label = '$^{18}$F[FDG] Tracer', linewidth = 2) # experimental data
plt.plot(tinterp[:,0]+deadVolumeTime[0], C_outNorm2[:,0],alpha = 0.6,color = "blue", label = 'G = (44,115)', linewidth = 2) # graph model
plt.plot(tinterp[:,1]+deadVolumeTime[1], C_outNorm2[:,1], color = "Orange", label = 'G = (16,36)', linewidth = 2) # graph model
plt.plot(tinterp[:,2]+deadVolumeTime[2], C_outNorm2[:,2],alpha = 0.6,color = "green", label = 'G = (8,16)', linewidth = 2) # graph model
# Format the plots
plt.legend(fontsize=27)
plt.xlim([0,1200])
plt.xlabel('Time (s)', fontsize = 32)
plt.ylabel('C/C$_0$ (-)', fontsize = 32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()

plt.figure(figsize=(10,6), dpi = 300)
plt.plot(trad3, granpoint5_rad3Norm, color = cmap(0.9005767012687428), label = '$^{18}$F[FDG] Tracer', linewidth = 2) # experimental data
plt.plot(tinterp[:,0]+deadVolumeTime[0], C_outNorm2[:,0], alpha = 0.6,color = "blue", label = 'G = (44,115)', linewidth = 2) # graph model
plt.plot(tinterp[:,1]+deadVolumeTime[1], C_outNorm2[:,1], color = "Orange", label = 'G = (16,36)', linewidth = 2) # graph model
plt.plot(tinterp[:,2]+deadVolumeTime[2], C_outNorm2[:,2], alpha = 0.6,color = "green", label = 'G = (8,16)', linewidth = 2) # graph model
# Format the plots
plt.xlim([0,1200])
plt.ylim([9E-3,0.5])
plt.yscale('log')
plt.xlabel('Time (s)', fontsize = 32)
plt.ylabel('C/C$_0$ (-)', fontsize = 32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()


### residuals for quantile comparison
residuals = np.zeros_like(quantileCalcsModel)
colors = ['blue', 'Orange', 'green']
labels = ['G = (44,115)', 'G = (16,36)', 'G = (8,16)']
plt.figure(figsize=(10,6), dpi = 300)
for i in range(len(graph_filename)):
    for j in range(len(residuals[0,:])):
        residuals[i,j] =  (quantileCalcsExp[i,j] - quantileCalcsModel[i,j])/quantileCalcsExp[i,j]
        
    plt.plot(quantiles, residuals[i,:]*100, color = colors[i], label = labels[i], linewidth = 3) # graph model

# Format the plot
plt.xlabel('Quantile', fontsize = 28)
plt.ylabel('Percent Error (%)', fontsize = 28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()




















