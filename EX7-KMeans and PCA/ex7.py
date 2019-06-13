import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from sklearn.cluster import KMeans
from ex7modules import *
#Part 1
X=loadmat('ex7data2.mat')['X']
K=3
max_iters=10
init_centroids=np.array([[3,3],[6,2],[8,5]])
centroids,idx=MyKMeans(X,init_centroids,max_iters,True)

#Part 2:Image Compression
fig=loadmat('bird_small')['A']/255   #Normalize
fig_size=fig.shape[0]
plt.imshow(fig)
plt.show()

fig=fig.reshape(3,-1).T  #convert(128,128,3) to (3,128*128) ,to fit the KMeans function

K=16  #reduce nmuber of colors to 16
max_iters=10

time_start=time.time()
init_centroids=InitCentroids(fig,K)

fig_centroids,_=MyKMeans(fig,init_centroids,max_iters,False) # find the most used K colors

fig_idx=findClosestCentroids(fig,fig_centroids)  # find every pixel's closest color
time_end=time.time()

print("Using my Kmeans costs time: ",time_end-time_start)

fig_recovered=np.zeros((fig_size*fig_size,3))  #assign every pixel to closest color
for i in range(fig_size*fig_size):
    fig_recovered[i,:]=fig_centroids[fig_idx[i]-1,:]

fig_recovered=fig_recovered.T.reshape((fig_size,fig_size,3))  #need to Transpose first,otherwise
                                                              #there is mistake in image
plt.imshow(fig_recovered)
plt.show()

#Part 3:Using SKlearn
time_start=time.time()
clf=KMeans(n_clusters=16,init='random',max_iter=50)
clf.fit(fig)
time_end=time.time()

print("Using SKlearn Kmeans costs time: ",time_end-time_start)

cluster_centers=clf.cluster_centers_
labels=clf.labels_

fig_recovered=np.zeros((fig_size*fig_size,3))
for i in range(fig_size*fig_size):
    fig_recovered[i,:]=cluster_centers[labels[i],:]

fig_recovered=fig_recovered.T.reshape((fig_size,fig_size,3))

plt.imshow(fig_recovered)
plt.show()


