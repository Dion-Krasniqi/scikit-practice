from sklearn.linear_model import Ridge

X = [[0,0], [0, 0], [1, 1]] # features wich are highly correlated, like in this case they are the same
                            # Standard regression model, like linear regression would struggle to see
                            # how each independent variable influences y. So ridge tries to do better
                            # by also minimizing the coefficients, or rather punishing large coefficients
                            # by multiplying each coefficient squared with alpha, and addign the sum
                            # to the Sres. In short, it tries to stop the model from overfitting, lets
                            # say a point dramatically can change the predicted line so the graph can go
                            # closer to that point, ridge balances it and builds a more generalized model
y = [0, 0.1, 1] # target values

reg = Ridge(alpha=.5) # The "penalty" (0,+inf), when 0 bascially linear regression, as it increases it forces
                      # coefficients to trend towards 0, inf means basically all 0

reg.fit(X, y)  # Trained model

print(reg.coef_) # coefficients of the model

print(reg.intercept_) # intercept of the model
