import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import accuracy, precision, recall

from tree.randomForest import RandomForestClassifier

###Write code here

np.random.seed(42)
n_estimators = 3
max_depth = 4

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

X = pd.DataFrame(X)
y = pd.Series(y, dtype='category')

for criteria in ['entropy', 'gini']:
    Classifier_RF = RandomForestClassifier(n_estimators, criterion=criteria, max_depth=max_depth)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    [fig1, fig2, fig3] = Classifier_RF.plot(X, y)
    
    fig1.savefig('plots/q5_classification_'+criteria+'_fig1.png', dpi=600, bbox_inches='tight')
    fig2.savefig('plots/q5_classification_'+criteria+'_fig2.png', dpi=600, bbox_inches='tight')
    fig3.savefig('plots/q5_classification_'+criteria+'_fig3.png', dpi=600, bbox_inches='tight')
    
    plt.show()
    print('\nCriteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in sorted(y.unique()):
        print(cls)
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))
