import numpy as np
import matplotlib.pyplot as plt

def LoadData(filename):
 with open(filename) as f:
    lines = f.readlines()
    n=len(lines[0].split(','))
    Data=[]
    for line in lines:
        data = line.split(',')
        data[n-1] = data[n-1].replace("\n", '')
        for i in range(n):
          data[i]=float(data[i])
        Data.append(data)
    Data=np.array(Data)
    X=Data[:,0:n-1]
    Y=Data[:,n-1]
    return X,Y

def DataReshape(X,Y):
    m = np.size(Y)
    VecOnes = np.ones((m, 1))
    X = np.column_stack((VecOnes, X))
    Y = Y.reshape((m, 1))
    return X,Y

def PlotData(X,Y):
    plt.xlabel('Population of City in 10,1000s')
    plt.ylabel('Profit in $10,1000s')
    plt.scatter(X, Y, c='red', marker='x',label='Training data')
    plt.show()


def computeCost(X,Y,theta):
    m=np.size(Y)
    J = np.sum(np.square(np.dot(X,theta) - Y)) / (2 * m)
    return J

def gradientDescent(X,Y,theta,alpha,num_iters):
    m=np.size(Y)
    J_history=np.zeros((num_iters,1))

    for i in range(num_iters+1):
        temp0=theta[0]-alpha/m*(np.sum(np.dot(X,theta)-Y))
        temp1=theta[1]-alpha/m*np.sum(np.dot((np.dot(X,theta)-Y).T,X[:,1]))
        theta[0]=temp0
        theta[1]=temp1
        J_history[i-1]=computeCost(X,Y,theta)
    return theta,J_history

def PlotLinear(X,Y,theta):
    X_p=X[:,1]
    plt.plot(X[:,1],np.dot(X,theta),'b-')
    PlotData(X_p, Y)


def PlotCost(J_history,num_iters):
    plt.xlabel('Number of iteration')
    plt.ylabel('Cost')
    plt.title('Cost Change')
    iters=np.arange(1,num_iters+1)
    plt.plot(iters,J_history)
    plt.show()

def Predict(X,theta):
    predict = np.dot(X, theta)
    print('for Population size %d,we predict a profit of %f' %(X[1]*10000,predict*10000))

def featureNormalize(X):
    n=np.size(X,1)
    X_norm=X
    mu=np.zeros((n,1))
    sigma=np.zeros((n,1))
    for i in range(1,n):
        mu[i]=np.mean(X[:,i])
        sigma[i]=np.std(X[:,i])
        X_norm[:,i]=(X[:,i]-mu[i])/sigma[i]
    X_norm[:,0]=1
    return X_norm,mu,sigma

def gradientDescentMulti(X,Y,theta,alpha,num_iters):
    m=len(Y)
    J_history=np.zeros((num_iters,1))
    for i in range(num_iters):
        temp0=theta[0]-alpha*np.sum(np.dot(X,theta)-Y)/m
        temp1=theta[1]-alpha*np.dot(X[:,1].T,np.dot(X,theta)-Y)/m
        temp2=theta[2]-alpha*np.dot(X[:,2].T,np.dot(X,theta)-Y)/m
        theta[0]=temp0
        theta[1]=temp1
        theta[2]=temp2
        J_history[i]=computeCost(X,Y,theta)
    return theta,J_history

def normalEqn(X,Y):
    theta=np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),Y)
    return theta