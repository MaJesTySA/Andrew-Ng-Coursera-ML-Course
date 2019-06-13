import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def lrCostFunc(theta,X,y,lamda):
    m = X.shape[0]
    theta = theta.reshape((theta.shape[0], 1))
    J=np.sum(np.square(np.dot(X,theta)-y))/(2*m)+ \
      lamda*np.sum(np.square(theta[1:]))/(2*m)
    return J

def lrgradient(theta,X,y,lamda):
    m = X.shape[0]
    theta = theta.reshape((theta.shape[0], 1))
    grad=np.dot(X.T, X.dot(theta) - y) / m + theta * lamda / m
    grad[0] = grad[0] - theta[0] * lamda / m
    return grad.flatten()

def learningCurve(X,y,Xval,yval,init_theta,lamda):
    train_error=np.zeros(X.shape[0])
    val_error=np.zeros(X.shape[0])
    X=np.column_stack((np.ones((X.shape[0],1)),X))
    for i in range(X.shape[0]):
        results = op.fmin_cg(f=lrCostFunc, x0=init_theta, disp=False,\
                             args=(X[:i+1,:], y[:i+1,:],lamda), fprime=lrgradient,maxiter=400)
        theta = results.reshape((X.shape[1], 1))
        train_error[i]=lrCostFunc(theta,X[:i+1,:],y[:i+1,:],lamda)
        val_error[i]=lrCostFunc(theta,np.column_stack((np.ones(Xval.shape[0]),Xval)),yval,lamda)
    return train_error,val_error

def polyFeatures(X,p):
    X_p = np.zeros((X.shape[0], p))
    for i in range(p):
        X_p[:, i] = np.power(X.flatten(), i + 1)
    return X_p

def featureNormalize(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0,ddof=1)
    X_norm = (X - mu) / sigma
    return X_norm,mu,sigma

def plotData(X,y):
    plt.plot(X, y, 'rx')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()

def plotLinearRegression(X,y,theta):
    Xone=np.column_stack((np.ones((X.shape[0],1)),X))
    plt.plot(X, y, 'rx')
    plt.plot(X, Xone.dot(theta), 'b')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()

def plotLearningCurve(train_error,val_error):
    l1, = plt.plot(np.arange(1, 13), train_error, 'b', label='train')
    l2, = plt.plot(np.arange(1, 13), val_error, 'g', label='cv')
    plt.ylim(0, 150)
    plt.legend(handles=[l1, l2], loc=1)
    plt.title("Learning curve for linear regression")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.show()

def plotPolyRegression(X,y,mu,sigma,theta,p):
    plt.plot(X, y, 'rx')
    xmin = np.min(X) - 15
    xmax = np.max(X) + 25
    x = np.arange(xmin, xmax, .05)
    x_poly = polyFeatures(x, p)
    x_poly = (x_poly - mu) / sigma
    x_poly = np.column_stack((np.ones((x_poly.shape[0], 1)), x_poly))
    plt.plot(x, x_poly.dot(theta), '--b')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.xlim(-80, 80)
    plt.ylim(np.min(x_poly.dot(theta)) - 10, np.max(x_poly.dot(theta) + 20))
    plt.show()

def plotLambdaError(lamda_vec,train_error,val_error):
    l1,=plt.plot(lamda_vec,train_error,'b',label='Train')
    l2,=plt.plot(lamda_vec,val_error,'g',label='Cross Validation')
    plt.legend(handles=[l1,l2],loc=1)
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.xlim(0,10)
    plt.ylim(0,25)
    plt.show()


def validationCurve(X, y, Xval, yval,init_theta):
    lamda_vec=np.array([0,0.001,0.003,0.01,0.03,.1,.3,1,3,10])
    train_error=np.zeros((lamda_vec.shape[0],1))
    val_error = np.zeros((lamda_vec.shape[0],1))
    i=-1
    for lamda in lamda_vec:
        i=i+1
        theta=op.fmin_cg(f=lrCostFunc,x0=init_theta,disp=False,\
                         args=(X,y,lamda),fprime=lrgradient,maxiter=200)
        train_error[i]=lrCostFunc(theta,X,y,0)
        val_error[i]=lrCostFunc(theta,Xval,yval,0)
    return lamda_vec,train_error,val_error