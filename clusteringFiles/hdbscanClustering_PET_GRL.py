# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:19:34 2023

@author: Collin Sutton
prepared for "A laboratory-validated, graph-based flow and transport model for naturally fractured media" 
submitted to Geophysical Research Letters
Reuse of this code must cite the original paper

"""

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics.pairwise import euclidean_distances

graph_filename = "PETGraph_cMax_point1" #this is using C Max from PET for each voxel as array position 3


dataFull = np.loadtxt(graph_filename + '.txt', delimiter=' ')
data = dataFull[:,0:-1]

fig = plt.figure(figsize=(18,6),dpi=300)
# ax = Axes3D(fig)
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0], data[:,2], data[:,1], s=5)
ax.set_aspect('equal')
# ax.view_init(azim=200)
plt.show()

#%%
import hdbscan
def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()


###min_cluster_size is the minimum number of samples in a group for that group to be considered a cluster. Clusters smaller than the ones of this size will be left as noise. The default value is 5. 
###min_samples is the number of samples in a neighborhood for a point to be considered as a core point, including the point itself. min_samples defaults to min_cluster_size.
######****Change PARAM = ({"min_cluster_size": 17, "min_samples": 17}) to change graph size********########
######****17 corresponds to the least complex graph, 15 corresponds to the middle graph, and 10 corresponds to the most complex*********###########

PARAM = (
    {"min_cluster_size": 17, "min_samples": 17})
hdb = HDBSCAN(**PARAM).fit(data)
labels = hdb.labels_
##plot without "noise" values
clusters = labels

fig = plt.figure(figsize=(18,6),dpi=300)
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(data[:,0], data[:,2], data[:,1], c=labels, s=3, cmap="viridis")
ax.set_aspect('equal')
plt.colorbar(sc, label="Cluster number")
plt.xlabel("\nCore width")
plt.ylabel("\nCore width")
ax.set_zlabel('\nPosition along core axis')
plt.show()

graphWithClusters = np.column_stack((data[:,0],data[:,1],data[:,2],clusters,dataFull[:,-1])) #x,y,z,cluster id, aperture
graphWithClusters_noNoise = graphWithClusters[(graphWithClusters >= 0).all(axis=1)]


fig = plt.figure(figsize=(18,6),dpi=300)
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(graphWithClusters_noNoise[:,0], graphWithClusters_noNoise[:,2], graphWithClusters_noNoise[:,1], c=graphWithClusters_noNoise[:,3], s=3, cmap="viridis")
plt.colorbar(sc, label="Cluster number")
plt.xlabel("\nCore width [cm]")
plt.ylabel("\nCore width [cm]")
ax.set_aspect('equal')
ax.set_zlabel('\nCore axis [cm]')
plt.show()

# # get centroid of each cluster
clusterData = graphWithClusters_noNoise
clusterData = graphWithClusters_noNoise[:-1]
points_of_cluster = {}
centroid_of_cluster = {}
for i in range(len(np.unique(clusterData[:,3]))):
    points_of_cluster[i] = clusterData[(clusterData[:,3] == i),:]
    centroid = np.mean(points_of_cluster[i], axis=0)
    centroid_of_cluster[i] = centroid

keys = list(centroid_of_cluster.keys())
lists = list(centroid_of_cluster.values())
#initialize varibles to hold center point data
x,y,z,cl,ap = [np.zeros(len(centroid_of_cluster)) for i in range(5)]

fig = plt.figure(figsize=(18,6),dpi=300)
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(graphWithClusters_noNoise[:,0], graphWithClusters_noNoise[:,2], graphWithClusters_noNoise[:,1], c=graphWithClusters_noNoise[:,3], s=.5,cmap="viridis", alpha=0.3)
for i in range(len(centroid_of_cluster)):
    x[i],y[i],z[i],cl[i],ap[i] = lists[i]
    ax.scatter(x[i], z[i], y[i], c="black", s=20)
plt.colorbar(sc, label="Cluster number")
plt.xlabel("Core Face (cm)")
plt.ylabel("\nDistance from inlet (cm)")
ax.set_zlabel('')
ax.set_aspect('equal')
ax.view_init(-140, 60)
plt.show()

PETgraphFull = np.column_stack((x[:],y[:],z[:],cl[:], ap[:])) #x,y,z,cluster id, max concentration
############end CT##############

###save for graph
graph_filename = "granite_frac_graphFromPET_cMax_HDBSCAN_minClusterSize_17"
# np.savetxt(graph_filename+'.txt', PETgraphFull) #x,y,z,cluster id, aperture

print("number of cluster found: {}".format(len(set(labels))))
print('cluster for each point: ', labels)




