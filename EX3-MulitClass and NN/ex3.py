from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(x):
    return 1/(1+np.exp(-x))

def lrCostFunc(theta,X,y):
    m = X.shape[0]
    lamda = 3
    theta = theta.reshape((theta.shape[0], 1))
    J=np.sum(-y*np.log(sigmoid(X.dot(theta)))-(1-y)*np.log(1-sigmoid(X.dot(theta))))/m+ \
      lamda*np.sum(np.square(theta[1:]))/(2*m)
    return J

def lrgradient(theta,X,y):
    m = X.shape[0]
    lamda = 3
    theta = theta.reshape((theta.shape[0], 1))
    grad = np.dot(X.T, sigmoid(X.dot(theta)) - y) / m + theta * lamda / m
    grad[0] = grad[0] - theta[0] * lamda / m
    return grad.flatten()

def predict(theta,X):
    y_hat=sigmoid(X.dot(theta.T))
    p=np.argmax(y_hat,axis=1)
    for i in range(p.shape[0]):
        if(p[i]==0):
            p[i]=10
    return p

data=loadmat('ex3data1.mat')
X=data['X']
y=data['y']

#show img
i=1000
X_re=X[1000].reshape([20,20])
plt.imshow(X_re.T,cmap='gray')
plt.show()

theta_t=np.array([-2,-1,1,2]).flatten()
X_t=np.column_stack((np.ones((5,1)),np.arange(.1,1.6,.1).reshape((3,5)).T))
y_t=np.array([1,0,1,0,1]).reshape((5,1))

#Check cost function and gradients function
J=lrCostFunc(theta_t,X_t,y_t)
grad=lrgradient(theta_t,X_t,y_t)
print('Cost: %f  (should be about 2.5348)'% J)
print('Gradients: [%f %f %f %f] \
(should be about [0.146561 -0.548558 0.724722 1.398003])' %(grad[0],grad[1],grad[2],grad[3]))

X=np.column_stack((np.ones(X.shape[0]),X))
theta=np.zeros(X.shape[1])
results=np.zeros((10,401))

#one Vs all
for k in range(1,11):
  if(k!=10):
      result=op.minimize(fun=lrCostFunc,x0=theta,args=(X,(y==k)+0),method='TNC',jac=lrgradient)
      results[k,:]=result['x']
  else:
      result = op.minimize(fun=lrCostFunc, x0=theta, args=(X, (y == k) + 0), method='TNC', jac=lrgradient)
      results[0, :] = result['x']

p=predict(results,X)
print("traing accuracy:",np.mean((y.flatten()==p)+0))





