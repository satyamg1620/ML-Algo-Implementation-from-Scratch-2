import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt

from metrics import rmse, mae

from ensemble.gradientBoosted import GradientBoostedRegressor

# Or use sklearn decision tree
from sklearn.tree import DecisionTreeRegressor

class DecisionTree():
    """
    Wrapper class for sklearn DT that returns pd.Series for predictions
    """
    def __init__(self, criterion='squared_error', max_depth=None):
        self.model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return pd.Series(self.model.predict(X))

########### GradientBoostedRegressor ###################

from sklearn.datasets import make_regression
 
n_estimators = 100
learning_rate = 0.1
max_depth = 2

X, y= make_regression(
    n_features=3,
    n_informative=3,
    noise=10,
    tail_strength=10,
    random_state=42,
)
 
X = pd.DataFrame(X)
y = pd.Series(y, name='y')

tree = DecisionTree(max_depth=max_depth)
Regressor_GB = GradientBoostedRegressor(tree, n_estimators=n_estimators, learning_rate=learning_rate)
Regressor_GB.fit(X, y)
y_hat = Regressor_GB.predict(X)
y_hat_ub = Regressor_GB.models[0].predict(X)

print('Unboosted:')
print('RMSE: ', rmse(y_hat_ub, y))
print('MAE: ', mae(y_hat_ub, y))

print("Gradient Boosted:")
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))

