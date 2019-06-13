from ex2modules import *
import numpy as np
import scipy.optimize as op


dgree=6
X, y = ReadData('ex2data2.txt')
X=mapFeatures(X,6)
theta=np.zeros(28)
result=op.minimize(fun=costFuncReg,x0=theta,args=(X,y),method='TNC',jac=gradientReg)
print(result)




