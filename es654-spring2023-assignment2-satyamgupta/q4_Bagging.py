#!/bin/python3

#%% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import accuracy, precision, recall
from ensemble.bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier # Use sklearn decision tree
np.random.seed(42)

#%% Wrapper class for sklearn DT that returns pd.Series for predictions

class skDecisionTree():
    def __init__(self, criterion):
        self.model = DecisionTreeClassifier(criterion=criterion)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return pd.Series(self.model.predict(X))

#%% BaggingClassifier

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

criteria = "entropy"
tree = skDecisionTree(criterion=criteria)

Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, n_jobs=1)
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)

[fig1, fig2] = Classifier_B.plot(X, y)

fig1.savefig('plots/q4_bagging_fig1.png', dpi=600, bbox_inches='tight')
fig2.savefig('plots/q4_bagging_fig2.png', dpi=600, bbox_inches='tight')

plt.show()

print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print(f"{cls} Precision: ", precision(y_hat, y, cls))
    print(f"{cls} Recall: ", recall(y_hat, y, cls))
    
#%% Mean accuracy vs n_estimators

n_iterations = 20
n_estimators_list = list(range(1, 50, 1))
accuracies = np.zeros((len(n_estimators_list), n_iterations))

for j in range(n_iterations):
    for i, n_estimators in enumerate(n_estimators_list):
        print(i, j)
        Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, n_jobs=1)
        Classifier_B.fit(X, y)
        y_hat = Classifier_B.predict(X)
        accuracies[i][j] = accuracy(y_hat, y)
np.save('data/q4_accuracies.npy', accuracies)

#%% Plot mean accuracy vs n_estimators

# accuracies = np.load('data/q4_accuracies.npy')
plt.errorbar(n_estimators_list, accuracies.mean(1), yerr=accuracies.std(1), fmt='-', elinewidth=1)
plt.xlabel("n_estimators")
plt.ylabel("mean_accuracy")
plt.savefig("plots/q4_accuracy.png", dpi=600, bbox_inches='tight')
plt.show()

#%% Timing Analysis
from timeit import timeit

n_iterations = 10
n_jobs_list = [1, 2, 4, 8]
n_estimators_list = [128, 256, 512, 1024]

times = np.zeros((len(n_jobs_list), len(n_estimators_list), 2))

for i, n_jobs in enumerate(n_jobs_list):
    for j, n_estimators in enumerate(n_estimators_list):
        print(f"\rn_jobs={n_jobs}, n_estimators={n_estimators}   ", end='')
        Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, n_jobs=n_jobs)
        times[i, j, 0] = timeit('Classifier_B.fit(X, y)', number=10, globals=globals())
        times[i, j, 1] = timeit('Classifier_B.predict(X)', number=10, globals=globals())
np.save("data/q4_times.npy", times)

#%% Plot timings

# times = np.load("data/q4_times.npy")

for i, n_jobs in enumerate(n_jobs_list):
    plt.plot(n_estimators_list, times[i, :, 0])

plt.xticks(n_estimators_list)
plt.legend([f"n_jobs={j}" for j in n_jobs_list])
plt.ylabel("fit time")
plt.xlabel("n_estimators")
plt.savefig("plots/q4_fit_times.png", dpi=600, bbox_inches="tight")
plt.show()

for i, n_jobs in enumerate(n_jobs_list):
    plt.plot(n_estimators_list, times[i, :, 1])

plt.xticks(n_estimators_list)
plt.legend([f"n_jobs={j}" for j in n_jobs_list])
plt.ylabel("predict time")
plt.xlabel("n_estimators")
plt.savefig("plots/q4_predict_times.png", dpi=600, bbox_inches="tight")
plt.show()