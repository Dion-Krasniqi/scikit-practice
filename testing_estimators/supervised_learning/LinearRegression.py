import numpy as np
from sklearn.linear_model import LinearRegression


# Two features/dimesion, each column is value for one of the features
X = np.array([[1, 1],
             [1, 2],
             [2, 2],
             [2, 3]])
print(X)

y = np.dot(X, np.array([1, 2])) + 3 # y is expected to be a linear combination of the input, so it is dependent and 
                                    # we are the actual target variables which are going to be
                                    # y = x1*1 + x2*2 + 3, so 3 is the intercept, x1 is the first feature 
                                    # x2 is the second, so for [2,3]: y = 2*1 + 3*2 + 3 = 11
                                    
                                    # To clarify y is a matrix product of [1,2] and X, so its basically all the target(actual) values

reg = LinearRegression().fit(X, y) # LinearRegression() blank model which knows Linear Regression, .fit starts the teaching and
                                   # stores the formula in reg
                                   
print(reg.score(X, y))      # makes prediction, checks what the actual results will be and judges how good are the 
# coefficients(1,2)         # predictions, best is 1(which is our case) and can be negative. Uses R squared which 
print(reg.coef_)            # is 1 - (Sres)/(Stot); If yi are the actual results, fi is the predicted => Sres = Σ[(yi-fi)**2]
# interecpt(3)              #                                                                            Stot = Σ[(yi-avg(yi))**2]
print(reg.intercept_)


# predict for x1 = 3 and x2 = 5, predicted value will be 3*1 + 5*2 + 3 = 16
print(reg.predict(np.array([[3,5]])))

# This particular model tries to minimize the Sres, so try to reduce the distance between the actual value and predicted one,
# if there is an error at the i-th value the result of the i-th is positive, if a prediction is off by a lot it gets
# much bigger due to the squaring.