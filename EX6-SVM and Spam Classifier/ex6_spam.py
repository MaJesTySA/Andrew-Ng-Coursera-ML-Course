from ex6spamModules import *
from scipy.io import loadmat
from sklearn.svm import SVC

x_emailSample1=EmailToFeatures('PYemailSample1.txt')
x_emailSample2=EmailToFeatures('PYemailSample2.txt')
x_spamSample1=EmailToFeatures('PYspamSample1.txt')
x_spamSample2=EmailToFeatures('PYspamSample2.txt')

dataTrain=loadmat('spamTrain.mat')
dataTest=loadmat('spamTest.mat')
X=dataTrain['X']
y=dataTrain['y'].ravel()

Xtest=dataTest['Xtest']
ytest=dataTest['ytest'].ravel()
clf=SVC(C=0.1,kernel='linear')
clf.fit(X,y)

print("Traning Accuracy: ",clf.score(X,y))
print("Test Accuracy: ",clf.score(Xtest,ytest))

print(clf.predict(x_emailSample1))
print(clf.predict(x_emailSample2))
print(clf.predict(x_spamSample1))
print(clf.predict(x_spamSample2))




