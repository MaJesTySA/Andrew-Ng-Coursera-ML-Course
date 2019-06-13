import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

def plotData(X,y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o', edgecolors='k')

def visualizeBoundary(X,y,clf):
    plotData(X,y)
    x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = clf.predict(this_X)
    plt.contour(X1, X2, vals,[0.5],colors='blue')
    plt.show()

def visualizeBoundaryLinear(X,y,clf):
    W=clf.coef_
    b=clf.intercept_
    xp=np.linspace(np.min(X[:,0]),np.max(X[:,1]),100)
    yp=(W[0][0]*xp+b)/(-1*W[0][1])
    plotData(X,y)
    plt.plot(xp,yp,c='b',linewidth=0.5)
    plt.show()

def gaussianKernel(x1,x2,sigma):
    return np.exp(-np.sum(np.square(x1-x2))/(2*sigma**2))

def findBest(Csteps,gammasteps,X,y,Xval,yval):
    errors = np.zeros((Csteps.shape[0], gammasteps.shape[0]))
    for Cstep in Csteps:
        for gammastep in gammasteps:
            clf = SVC(C=Cstep, kernel='rbf', gamma=gammastep)
            clf.fit(X, y)
            errors[np.where(Csteps == Cstep), np.where(gammasteps == gammastep)] = 1 - clf.score(Xval, yval)
    idx = np.argmin(errors)
    i = int(idx / Csteps.shape[0])
    j = idx - i * Csteps.shape[0]
    Cmin = Csteps[i]
    gammamin = gammasteps[j]
    return Cmin,gammamin