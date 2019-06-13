import scipy.optimize as op
import numpy as np
from scipy.io import loadmat
from ex5modules import *

#Part 1: Loading and visualize data
data=loadmat('ex5data1.mat')
X=data['X']
y=data['y']
Xtest=data['Xtest']
ytest=data['ytest']
Xval=data['Xval']
yval=data['yval']

plotData(X,y)

#Part 2: Costs and gradients
Xone=np.column_stack((np.ones((X.shape[0],1)),X))
theta_t=np.array([1,1]).flatten()

cost=lrCostFunc(theta_t,Xone,y,1)
grad=lrgradient(theta_t,Xone,y,1)
print("Cost at theta_t=[1,1] with lambda=1: %f  (should be about 303.993)"%cost)
print("Gradient at theta_t=[1,1] with lambda=1: [%f,%f]  (should be about [-15.30,598.250])"%(grad[0],grad[1]))

#Part 3: Train Linear Regression
init_theta=np.zeros(Xone.shape[1])
lamda=0

theta=op.fmin_cg(f=lrCostFunc,x0=init_theta,args=(Xone,y,lamda),\
                 fprime=lrgradient,disp=False).reshape((Xone.shape[1],1))
plotLinearRegression(X,y,theta)

#Part 4: Learning curve for Linear Regression
train_error,val_error=learningCurve(X,y,Xval,yval,init_theta,lamda)
plotLearningCurve(train_error,val_error)

#Part 5: Feature Mapping for Polynomial Regression
p=8

X_poly=polyFeatures(X,p)
X_poly,mu,sigma=featureNormalize(X_poly)
X_poly_one=np.column_stack((np.ones((X_poly.shape[0],1)),X_poly))

X_poly_test=polyFeatures(Xtest,p)
X_poly_test=(X_poly_test-mu)/sigma
X_poly_test_one=np.column_stack((np.ones((X_poly_test.shape[0],1)),X_poly_test))

X_poly_val=polyFeatures(Xval,p)
X_poly_val=(X_poly_val-mu)/sigma
X_poly_val_one=np.column_stack((np.ones((X_poly_val.shape[0],1)),X_poly_val))

lamda=100
poly_init_theta=np.zeros(X_poly_one.shape[1])

theta=op.fmin_cg(f=lrCostFunc,x0=poly_init_theta,args=(X_poly_one,y,lamda),\
                 fprime=lrgradient,disp=False,maxiter=200).reshape((X_poly_one.shape[1],1))
plotPolyRegression(X,y,mu,sigma,theta,p)

#Part 6: Learning curve for Polynomial Regression
train_error,val_error=learningCurve(X_poly,y,X_poly_val,yval,poly_init_theta,lamda)
plotLearningCurve(train_error,val_error)

#Part 7: Validation for Selecting Lambda

lamda_vec,train_error,val_error=validationCurve(X_poly_one, y, X_poly_val_one, yval,poly_init_theta)
plotLambdaError(lamda_vec,train_error,val_error)

print("\nlambda\t\tTrain Error\tValidation Error")
for i in range(lamda_vec.shape[0]):
    print(' %f\t%f\t%f'%(lamda_vec[i], train_error[i], val_error[i]))

#Part 8: Testing error with best lambda=3
lamda=3
theta=op.fmin_cg(f=lrCostFunc,x0=poly_init_theta,args=(X_poly_one,y,lamda),fprime=lrgradient,disp=False,maxiter=200)
test_error=lrCostFunc(theta,X_poly_test_one,ytest,0)
print("Test error with lambda=3:  %f  (should be about 3.8599)" % test_error)

