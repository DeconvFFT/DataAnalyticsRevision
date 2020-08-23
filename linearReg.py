"""
    solving linear regression directly using matrix ops
"""
from  numpy import array
from numpy.linalg import inv
import matplotlib.pyplot as plt
data = array([
   [0.05, 0.12],
	[0.18, 0.22],
	[0.31, 0.35],
	[0.42, 0.38],
	[0.5, 0.49]
])

X,y = data[:,0],data[:,1]
X = X.reshape(len(X),1)

#  linear least squares
b = inv(X.T.dot(X)).dot(X.T).dot(y)
print(b)

#predict using coefficients
yhat = X.dot(b)
# plot data and coefficients
plt.scatter(X,y)
plt.plot(X,yhat,color="red")
plt.show()


""" solving using QR decomposition"""
from numpy.linalg import qr
X,y = data[:,0],data[:,1]
X = X.reshape(len(X),1)
Q,R = qr(X)
b = inv(R).dot(Q.T).dot(y)
print(b)

#predict using coefficients
yhat = X.dot(b)
#plot data and predictions
plt.scatter(X,y)
plt.plot(X,yhat, color = "red")
plt.show()


""" solve using svd numpy and pseudo inverse"""
from numpy.linalg import pinv
X,y = data[:,0],data[:,1]
X = X.reshape(len(X),1)
# calculate coefficients
b = pinv(X).dot(y)
print(b)

#predict using coefficients
yhat = X.dot(b)
#plot data and predictions
plt.scatter(X,y)
plt.plot(X,yhat,color="red")
plt.show()


""" solve using numpy lstsq """

from numpy.linalg import lstsq
import numpy as np
X,y = data[:,0],data[:,1]
X1 = np.vstack([X,np.ones(len(X))]).T

#we will calculate m,c using lstsq
m,c = lstsq(X1,y)[0]
print(m,c)
plt.scatter(X,y)
plt.plot(X,m*X+c,color="red")
plt.show()