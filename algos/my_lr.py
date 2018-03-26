'''
Implementation of linear regression using numpy. We make use of sweedish insurance data set. 
y=mx+b
m:slope 
b:intercept, it controls the starting point of the line where it intersects the y-axis.
'''
import numpy as np
dataset=np.loadtxt("insurance.csv",delimiter=",")
x=dataset[:,0]
y=dataset[0:,1]
mean_x=x.mean()
mean_y=y.mean()
var_x=x.var()
var_y=y.mean()
# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
print(covariance(x, mean_x, y, mean_y))
#finding the coefficients 
def coefficients(x,y,mean_x,mean_y,var_x):
	m=covariance(x, mean_x, y, mean_y)/var_x
	b=mean_y-(mean_x*m)
	print(m)
	print(b)
coefficients(x,y,mean_x,mean_y,var_x)