import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt

from metrics import accuracy, precision, recall
from ensemble.ADABoost import AdaBoostClassifier

# Or you could import sklearn DecisionTree
from sklearn.tree import DecisionTreeClassifier as DecisionTree

########### AdaBoostClassifier on Real Input and Discrete Output ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 6
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

criteria = "entropy"
tree = DecisionTree(max_depth=1, criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot(X, y)

fig1.savefig('plots/q3_adaboost_fig1.png', dpi=600, bbox_inches='tight')
fig2.savefig('plots/q3_adaboost_fig2.png', dpi=600, bbox_inches='tight')

plt.show()

print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))
    
#%% Compare to sklearn    

print("\nCompare to scikit-learn...")
from sklearn.ensemble import AdaBoostClassifier as ABC
clsAbc = ABC(estimator=tree, n_estimators=n_estimators, algorithm="SAMME")
clsAbc.fit(X, y)
y_hat = clsAbc.predict(X)

print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))
    
#%% Compare to single decision stump

print("\nCompare to single decision stump...")
tree.fit(X, y)
y_hat = tree.predict(X)

print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))
