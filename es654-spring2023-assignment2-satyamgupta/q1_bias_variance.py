import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor as DecisionTree

np.random.seed(1234)

N = 5000

x = np.linspace(0, 10, N)
eps = np.random.normal(0, 5, N)
y = x**2 + 1 + eps

X_train = pd.DataFrame(x, columns=['x'])
y_train = pd.Series(y, name='y')

x = np.linspace(10, 11, N)
eps = np.random.normal(0, 5, N)
y = x**2 + 1 + eps

X_test = pd.DataFrame(x, columns=['x'])
y_test = pd.Series(y, name='y')

df_train = pd.concat([X_train, y_train], axis=1)

n_trials = 200
maximum_depth = 6
depths = list(range(1, maximum_depth + 1))

bias_squared = np.zeros(maximum_depth)
variance = np.zeros(maximum_depth)

for i, d in enumerate(depths):
    y_hats = np.zeros((len(X_test), n_trials))
    for j in range(n_trials):
        tree = DecisionTree(max_depth=d)
        bootstrap = df_train.sample(frac=1, replace=True)
        tree.fit(bootstrap.iloc[:,:-1], bootstrap.iloc[:,-1])
        y_hats[:, j] = tree.predict(X_test)
        
    bias_squared[i] = ((y_hats.mean(1) - y_test + eps)**2).mean()
    variance[i] = y_hats.var(1).mean()
    
#%%

np.save("data/q1_bias_squared_"+str(N)+".npy", bias_squared)
np.save("data/q1_variance_"+str(N)+".npy", variance)

plt.plot(depths, bias_squared, 'r-')
plt.plot(depths, variance, 'g-')
plt.legend(["bias_squared", "variance"])
plt.xlabel("max_depth")
plt.title(r"$Bias^2$ and Variance vs max_depth")
plt.savefig("plots/q1_bias_variance_"+str(N)+".png", dpi=600, bbox_inches="tight")
plt.show()

bias_squared_normalized = bias_squared/bias_squared.max()
variance_normalized = variance/variance.max()

plt.plot(depths, bias_squared_normalized, 'r-')
plt.plot(depths, variance_normalized, 'g-')
plt.legend(["bias_squared", "variance"])
plt.xlabel("max_depth")
plt.title(r"$Bias^2$ and Variance (normalized) vs max_depth")
plt.savefig("plots/q1_bias_variance_normalized_"+str(N)+".png", dpi=600, bbox_inches="tight")
plt.show()