from ex2modules import *
import scipy.optimize as op

X, y = ReadData('ex2data1.txt')
PlotData(X,y)
m=X.shape[0]
X=np.column_stack((np.ones((X.shape[0],1)),X))

theta=np.zeros(3)
print(costFunc(theta,X,y))
result=op.minimize(fun=costFunc,x0=theta,args=(X,y),method='TNC',jac=gradient)
theta=result['x']




