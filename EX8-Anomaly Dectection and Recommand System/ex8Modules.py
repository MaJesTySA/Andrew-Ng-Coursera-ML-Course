import numpy as np
import math
import matplotlib.pyplot as plt

def estimateGaussian(X):
    (m,n)=X.shape
    mu=np.zeros((n,1))
    sigma2=np.zeros((n,1))
    for i in range(n):
        mu[i]=np.mean(X[:,i])
        sigma2[i]=np.sum(np.square(X[:,i]-mu[i]))/m
    return mu,sigma2

def multiGaussian(X,mu,sigma2):
    (n,m) = mu.shape
    mu = mu.reshape((m, n))
    if sigma2.shape[0] == 1 or sigma2.shape[1] == 1:
        sigma2 = np.diag(sigma2.ravel())
    p = (2 * math.pi) ** (-n / 2) * np.linalg.det(sigma2) ** (-0.5) \
        * np.exp(-0.5 * np.sum((X - mu).dot(np.linalg.pinv(sigma2)) * (X - mu), axis=1))
    return p

def visualize(X):
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='b')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()

def visualizeFit(X,mu,sigma2,outliers,flag):
    X1, X2 = np.meshgrid(np.arange(0, 35.5, .5), np.arange(0, 35.5, .5))
    XX = np.zeros((71 * 71, 2))
    for i in range(71):
        XX[i * 71:(i + 1) * 71, 0] = i * 0.5
        XX[i * 71:(i + 1) * 71, 1] = np.arange(0, 35.5, .5)
    Z = multiGaussian(XX, mu, sigma2).reshape(71,-1).T
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='b')
    if (np.sum(np.isinf(Z) == 0)):
        plt.contour(X1, X2, Z, 10.0 ** np.arange(-20, 0, 3))
    if(flag==1):
        plt.scatter(X[outliers[0],0],X[outliers[0],1],marker='x',c='r')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.xlim(0,30)
    plt.ylim(0,30)
    plt.show()

def selectThreshold(yval,pval):
    bestEpsilon = 0
    bestF1 = 0
    pmin = np.min(pval)
    pmax = np.max(pval)
    stepsize = (pmax - pmin) / 1000
    for epsilon in np.arange(pmin, pmax, stepsize):
        cvPredictions = np.zeros((pval.shape[0], 1))
        for i in range(pval.shape[0]):
            if pval[i] < epsilon:
                cvPredictions[i] = 1
        fp = np.sum((yval == 0) & (cvPredictions == 1) + 0)
        tp = np.sum((yval == 1) & (cvPredictions == 1) + 0)
        fn = np.sum((yval == 1) & (cvPredictions == 0) + 0)
        prec = tp / (tp + fp + 1e-20)
        rec = tp / (tp + fn + 1e-20)
        F1 = 2 * prec * rec / (prec + rec + 1e-20)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestF1,bestEpsilon

def cofiCostFunc(params,Y,R,num_users,num_movies,num_features,lamda):
    X=params[:num_movies*num_features].reshape((num_movies,num_features))
    Theta=params[num_features*num_movies:].reshape((num_users,num_features))
    J = np.sum(np.square(R * X.dot(Theta.T) - Y)) / 2 + lamda / 2 * np.sum(np.square(Theta)) \
        + lamda / 2 * np.sum(np.square(X))
    return J

def cofiGrads(params,Y,R,num_users,num_movies,num_features,lamda):
    X = params[:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_features * num_movies:].reshape((num_users, num_features))
    X_grads = ((R * X.dot(Theta.T) - Y).dot(Theta) + lamda * X).ravel('F')
    Theta_grads = ((R * X.dot(Theta.T) - Y).T.dot(X) + lamda * Theta).ravel('F')
    grads = np.r_[X_grads, Theta_grads]
    return grads.flatten()

def loadMovieList():
    with open('movie_ids.txt', 'rb') as f:
        lines = f.readlines()
        moviesList = []
        for line in lines:
            line = str(line).split(' ', 1)
            newline = line[1].replace('\\n', '')
            newline = newline.replace('\'', '')
            newline = newline.replace('\"', '')
            moviesList.append(newline)
    return moviesList

def normalizeRatings(Y,R):
    ymean = np.zeros((Y.shape[0], 1))
    ynorm = np.zeros(Y.shape)
    for i in range(R.shape[0]):
        idx = np.where(R[i, :] == 1)[0]
        numrated = len(idx)
        ymean[i] = np.sum(Y[i, :]) / numrated
        ynorm[i,idx] = Y[i, idx] - ymean[i]
    return ymean,ynorm

def findCloestMovies(my_ratings,X,movieList,num):
    j = 0
    myFavormovies=X[np.where(my_ratings>0)[0],:]
    recmvList = np.zeros((myFavormovies.shape[0], num))
    for movie in myFavormovies:
        dist = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            dist[i] = np.linalg.norm(movie - X[i, :])
        sorted = np.argsort((dist).ravel())[1:num+1]
        recmvList[j:] = sorted
        j = j + 1
    j = -1
    for i in range(len(movieList)):
        if my_ratings[i] > 0:
            j = j + 1
            idx = recmvList[j, :]
            print("%d movies similar to movie %s is" % (num,movieList[i]))
            for k in range(num):
              print("                         %d %s"%(k+1,movieList[int(idx[k])]))
