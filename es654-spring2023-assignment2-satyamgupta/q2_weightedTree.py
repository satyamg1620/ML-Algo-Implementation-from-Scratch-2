import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt

from tree.base import WeightedDecisionTree
from metrics import accuracy, precision, recall

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

n_samples = 100
max_depth = 100

# make classification dataset
X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, 
                           random_state=1, n_clusters_per_class=2, class_sep=0.5)

X = pd.DataFrame(X)
y = pd.Series(y, dtype="category")

# add weights to the samples
w = np.random.uniform(size=n_samples)
w = w / np.sum(w)
X['W'] = w

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.3, shuffle=True)

# remove the weights from the inputs
w_train = X_train.iloc[:,-1]
X_train = X_train.drop(columns=['W'])
X_test = X_test.drop(columns=['W'])

for criteria in ["information_gain", "gini_index"]:
    tree = WeightedDecisionTree(criterion=criteria, max_depth=max_depth)  # Split based on Inf. Gain
    tree.fit(X_train, y_train, sample_weight=w_train)
    y_hat = tree.predict(X_test)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y_test))
    for cls in y_train.unique():
        print("Precision: ", precision(y_hat, y_test, cls))
        print("Recall: ", recall(y_hat, y_test, cls))
    tree.plot_Decision_Boundary(X_train, y_train)
    plt.savefig("plots/q2_boundary_"+criteria+".png", dpi=600, bbox_inches="tight")
    plt.show()
    
# compare to sklearn
for criteria in ["entropy", "gini"]:
    sktree = DecisionTreeClassifier(criterion=criteria, max_depth=max_depth)
    sktree.fit(X_train, y_train, sample_weight=w_train)
    y_hat = sktree.predict(X_test)
    plot_tree(sktree, class_names=['0', '1'])
    plt.savefig("plots/q2_tree_"+criteria+"_sk.png", dpi=600, bbox_inches="tight")
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y_test))
    for cls in y_train.unique():
        print("Precision: ", precision(y_hat, y_test, cls))
        print("Recall: ", recall(y_hat, y_test, cls))
    disp = DecisionBoundaryDisplay.from_estimator(sktree, X_train, response_method='predict', cmap="Paired")
    
    # Plot the training points
    for class_value in range(2):
        row_ix = np.where(y_train == class_value)
        disp.ax_.scatter(X_train.to_numpy()[row_ix,0], X_train.to_numpy()[row_ix,1], cmap='Paired')
    plt.savefig("plots/q2_boundary_"+criteria+"_sk.png", dpi=600, bbox_inches="tight")
    plt.show()