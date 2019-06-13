from ex1modules import *
#initialize parameters
alpha=0.01
num_iters=400
theta=np.zeros((3,1))

#load Data
X,Y=LoadData('ex1data2.txt')
X,Y=DataReshape(X,Y)

#Using Normal Equation to predict
theta=normalEqn(X,Y)
price1=np.dot(np.array([1,1650,3]).T,theta)
print('Predicted price of a 1650 sq-ft, 3 br house using Normal Equation :%f\n' % price1 )

#Using Gradient Descent to predict
theta=np.zeros((3,1))
X_norm,mu,sigma=featureNormalize(X)
theta,J_history=gradientDescentMulti(X,Y,theta,alpha,num_iters)
PlotCost(J_history,num_iters)
price2=np.dot(np.array([1,(1650-mu[1])/sigma[1],(3-mu[2])/sigma[2]]).T,theta)
print('Predicted price of a 1650 sq-ft, 3 br house using Gradient Descent :%f\n' % price2 )