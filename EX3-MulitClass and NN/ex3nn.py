from scipy.io import loadmat
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def predictNN(theta1,theta2,X):
    a1=sigmoid(X.dot(theta1.T))
    a1=np.column_stack((np.ones(a1.shape[0]),a1))
    y_hat=sigmoid(a1.dot(theta2.T))
    p = np.argmax(y_hat, axis=1)
    return p

weights=loadmat('ex3weights.mat')
data=loadmat('ex3data1.mat')
X=data['X']
y=data['y']
theta1=weights['Theta1']
theta2=weights['Theta2']

X=np.column_stack((np.ones(X.shape[0]),X))

p=predictNN(theta1,theta2,X)
print("train accuracy:",np.mean((y.flatten()==p+1)+0))