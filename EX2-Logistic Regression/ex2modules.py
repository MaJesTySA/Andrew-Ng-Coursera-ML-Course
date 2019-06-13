import numpy as np
import matplotlib.pyplot as plt


def ReadData(filename):
    with open(filename) as f:
        lines = f.readlines()
        m = len(lines)
        X = np.zeros((m, 2))
        y = np.zeros((m, 1))
        i = 0
        for line in lines:
            line = line.split(',')
            X[i, 0] = float(line[0])
            X[i, 1] = float(line[1])
            y[i, 0] = int(line[2])
            i += 1
        return X,y

def PlotData(X,y):
    pos=np.where(y==1)
    neg=np.where(y==0)
    l1=plt.scatter(X[pos,0],X[pos,1],c='k',marker='+')
    l2=plt.scatter(X[neg,0],X[neg,1],c='y',marker='o',edgecolors='k')
    plt.legend([l1,l2],['Admitted', 'Not admitted'],loc=1)
    plt.show()
    return plt

def sigmoid(x):
    z=1/(1+np.exp(-x))
    return z

def costFunc(theta,X,y):
    m=X.shape[0]
    theta = theta.reshape((theta.shape[0], 1))
    J=np.sum(-y*np.log(sigmoid(X.dot(theta)))- \
        (1-y)*np.log(1-sigmoid(X.dot(theta))))/m
    return J

def gradient(theta,X,y):
    m=X.shape[0]
    theta = theta.reshape((theta.shape[0], 1))
    grad = np.dot(X.T, sigmoid(X.dot(theta)) - y) / m
    return grad.flatten()

def costFuncReg(theta,X,y):
    m=X.shape[0]
    lamda=1
    theta = theta.reshape((theta.shape[0], 1))
    J=np.sum(-y*np.log(sigmoid(X.dot(theta)))- \
        (1-y)*np.log(1-sigmoid(X.dot(theta))))/m \
        +np.sum(np.square(theta))*lamda/(2*m)
    return J

def gradientReg(theta,X,y):
    lamda=1
    m=X.shape[0]
    theta = theta.reshape((theta.shape[0], 1))
    grad = np.dot(X.T, sigmoid(X.dot(theta)) - y) / m \
                 +theta*lamda/m
    grad[0]=grad[0]-theta[0]*lamda/m
    return grad.flatten()

def mapFeatures(X,dgree):
    X_featured=np.zeros((X.shape[0],nplus(dgree)+dgree+1))
    for i in range(dgree+1):
        for j in range(i+1):
            X_featured[:,i+j]=np.power(X[:,0],i-j)*np.power(X[:,1],j)
    return X_featured

def nplus(n):
    if(n==1):
        result=1
    else:
        result = n+nplus(n-1)
    return result




def PlotDB(X,y,theta):
    pass
