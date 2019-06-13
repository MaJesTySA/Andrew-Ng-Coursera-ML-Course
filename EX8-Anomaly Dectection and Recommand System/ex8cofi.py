from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from ex8Modules import *

#Part 1: Loading movie ratings dataset
moviedata=loadmat('ex8_movies.mat')
Y=moviedata['Y']
R=moviedata['R']
ysum=np.sum(Y[0,:])
rsum=np.where(R[0,:]==1)[0].shape[0]
print("Average rating for movie 1 (Toy Story): %f /5\n" % (ysum/rsum))
plt.imshow(Y,aspect='auto')
plt.ylabel("Movies")
plt.xlabel("Users")
plt.show()

#Part 2: Collaborative Filtering Cost Function and Gradient
params=loadmat('ex8_movieParams.mat')
X=params['X']
Theta=params['Theta']
num_users=4
num_movies=5
num_features=3

X_test=X[:num_movies,:num_features]
Theta_test=Theta[:num_users,:num_features]
Y_test=Y[:num_movies,:num_users]
R_test=R[:num_movies,:num_users]

param=np.r_[X_test,Theta_test].ravel()

J=cofiCostFunc(param,Y_test,R_test,num_users,num_movies,num_features,lamda=0)
print("Cost at loaded parameters with lambda=0: %f (should be about 22.22)" % J)
J=cofiCostFunc(param,Y_test,R_test,num_users,num_movies,num_features,lamda=1.5)
print("Cost at loaded parameters with lambda=1.5: %f (should be about 34.34)" % J)

#Part 3: Ratings for a new user
movieList=loadMovieList()
my_ratings=np.zeros((len(movieList),1))
my_ratings[0]=4
my_ratings[97]=2
my_ratings[6]=3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5
print("\nNew user ratings: ")
for i in range(len(movieList)):
    if my_ratings[i]>0:
        print("Rated %d for %s" % (my_ratings[i],movieList[i]))

#Part 4: Learning Movie Ratings
Y=np.column_stack((my_ratings,Y))
R=np.column_stack(((my_ratings!=0)+0,R))
ymean,ynorm=normalizeRatings(Y,R)

num_users=Y.shape[1]
num_movies=Y.shape[0]
num_features=10
lamda=10

X=loadmat('X.mat')['X']
Theta=loadmat('Theta.mat')['Theta']
init_params=np.r_[X,Theta].ravel()
#X=np.random.randn(num_movies,num_features)*0.1
#Theta=np.random.randn(num_users,num_features)*0.1

print("\nTraining collaborative filtering...")
results=op.fmin_cg(f=cofiCostFunc,x0=init_params,\
                   args=(ynorm,R,num_users,num_movies,num_features,lamda),fprime=cofiGrads)

X=results[:num_movies*num_features].reshape((num_movies,num_features))
Theta=results[num_movies*num_features:].reshape((num_users,num_features))
print("Recommender system learning completed.")

#Part 5: Recommendation for you
p=X.dot(Theta.T)
my_pred=p[:,0]+ymean.ravel()
pred_idx=np.argsort(-my_pred)
pred_score=np.sort(-my_pred)

print("\nTop recommendations for you :")
for i in range(10):
    print('Predicting rating %.1f for movie %s' % (-pred_score[i],movieList[pred_idx[i]]))

print("\nOriginal user ratings: ")
for i in range(len(movieList)):
    if my_ratings[i]>0:
        print("Rated %d for %s" % (my_ratings[i],movieList[i]))

print()

#Part 6: Find closest movies
findCloestMovies(my_ratings,X,movieList,5)