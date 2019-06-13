from ex1modules import *

theta=np.zeros((2,1))
num_iters=1500
alpha=0.01

X,Y=LoadData('ex1data1.txt')

PlotData(X,Y)

#Reshape Data
X,Y=DataReshape(X,Y)

#Computer Cost
J=computeCost(X,Y,theta)
print("Cost at theta=[0;0] : %f  (should be about 32.07)" % J)

J=computeCost(X,Y,np.array([[-1],[2]]))
print("Cost at theta=[-1;2] : %f  (should be about 54.24)" % J)

#Computer Gradient and J_history
theta,J_history=gradientDescent(X,Y,theta,alpha,num_iters)
print("Theta found by gradient descent: [%f %f] (should be about [-3.6303 1.1664]  "%(theta[0],theta[1]))

PlotLinear(X,Y,theta)
PlotCost(J_history,num_iters)

predict1=Predict(np.array([1,3.5]),theta)
predict2=Predict(np.array([1,7]),theta)
