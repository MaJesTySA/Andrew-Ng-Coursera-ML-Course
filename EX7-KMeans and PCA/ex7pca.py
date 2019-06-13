from scipy.io import loadmat
from ex7modules import *
#part 1
X=loadmat('ex7data1.mat')['X']

X_norm,mu,sigma=featureNormalize(X)
U,S=PCA(X_norm)
visualizeEigVector(mu,U,S,X)

print('Top eigenvector: ')
print(U[0,0],U[0,1])
print()

K=1
Z=projectData(X_norm,U,K)
X_rec=recoverData(Z,U,K)
visualizePCA(X_norm,X_rec)

#Part 2
X=loadmat('ex7faces.mat')['X']
displayData(X,10)

X_norm,_,_=featureNormalize(X)
U,S=PCA(X_norm)
displayData(U.T,6)

K=100
Z=projectData(X_norm,U,K)
print('The projected data Z has a size of: ',Z.shape)
X_rec=recoverData(Z,U,K)
displayData(X_rec,10)

Cpcompare(X,X_rec,10)


