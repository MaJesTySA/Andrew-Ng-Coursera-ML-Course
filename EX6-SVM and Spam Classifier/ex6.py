import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from ex6modules import *
#Exmaple 1
data=loadmat('ex6data1.mat')
X=data['X']
y=data['y'].ravel()
plotData(X,y)
plt.show()

#Plot C=1's boundary
clf=SVC(C=1,kernel='linear')
clf.fit(X,y)
visualizeBoundaryLinear(X,y,clf)
#Plot C=100's boundary
clf=SVC(C=100,kernel='linear')
clf.fit(X,y)
visualizeBoundaryLinear(X,y,clf)

#Examing Gaussian Kernel
sim=gaussianKernel(np.array([1,2,1]),np.array([0,4,-1]),sigma=2)
print("similarity :",sim)

#Example 2
data=loadmat('ex6data2.mat')
X=data['X']
y=data['y'].ravel()
plotData(X,y)
plt.show()

clf=SVC(C=1,kernel='rbf',gamma=50) #gamma=1/(2*sigma**2)
clf.fit(X,y)
visualizeBoundary(X,y,clf)

#Exmaple 3
data=loadmat('ex6data3.mat')
X=data['X']
y=data['y'].ravel()
Xval=data['Xval']
yval=data['yval']
plotData(X,y)
plt.show()

Csteps=np.array([.01,.03,.1,.3,1,3,10,30])
gammasteps=np.array([1/(2*.01**2),1/(2*.03**2),1/(2*.1**2),1/(2*.3**2),\
                     1/(2*1**2),1/(2*3**2),1/(2*10**2),1/(2*30**2)])

Cmin,gammamin=findBest(Csteps,gammasteps,X,y,Xval,yval)

clf=SVC(C=Cmin,kernel='rbf',gamma=gammamin)
clf.fit(X,y)
visualizeBoundary(X,y,clf)
