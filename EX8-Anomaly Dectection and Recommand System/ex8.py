from scipy.io import loadmat
from ex8Modules import *
import numpy as np

#Part 1:low-dimensional exmaples
data=loadmat('ex8data1.mat')
X=data['X'] #X.shape (307,2)
Xval=data['Xval'] #Xval.shape (307,2)
yval=data['yval'] #yval.shape (307,1)

visualize(X)

mu,sigma=estimateGaussian(X)
visualizeFit(X,mu,sigma,0,0)

p=multiGaussian(X,mu,sigma)
pval=multiGaussian(Xval,mu,sigma)

_,epsilon=selectThreshold(yval,pval)
outliers=np.where(p<epsilon)
visualizeFit(X,mu,sigma,outliers,1)

#Part 2:High-dimensional exmaples
data=loadmat('ex8data2.mat')
X=data['X']
Xval=data['Xval']
yval=data['yval']

mu,sigma=estimateGaussian(X)
p=multiGaussian(X,mu,sigma)
pval=multiGaussian(Xval,mu,sigma)
F1,epsilon=selectThreshold(yval,pval)

print("Best epsilon found using cv-sets: ",epsilon)
print("Best F1 score on cv-sets: ",F1)
print("Outliers found:",np.sum(p<epsilon))
