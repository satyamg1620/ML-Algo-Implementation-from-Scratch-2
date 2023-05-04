import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import accuracy, precision, recall, rmse, mae

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)
n_estimators = 3
max_depth = 4

#%% RandomForestClassifier
print("## RandomForestClassifier ##")

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

for criteria in ['entropy', 'gini']:
    Classifier_RF = RandomForestClassifier(n_estimators, criterion=criteria, max_depth=max_depth)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    [fig1, fig2, fig3] = Classifier_RF.plot(X, y)
    
    fig1.savefig('plots/q5_RFC_'+criteria+'_fig1.png', dpi=600, bbox_inches='tight')
    fig2.savefig('plots/q5_RFC_'+criteria+'_fig2.png', dpi=600, bbox_inches='tight')
    fig3.savefig('plots/q5_RFC_'+criteria+'_fig3.png', dpi=600, bbox_inches='tight')
    
    plt.show()
    print('\nCriteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in sorted(y.unique()):
        print(cls)
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

#%% RandomForestRegressor
print("\n## RandomForestRegressor ##")

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

Regressor_RF = RandomForestRegressor(n_estimators, max_depth=max_depth)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
[fig1, fig2, fig3] = Regressor_RF.plot(X, y)

fig1.savefig('plots/q5_RFR_fig1.png', dpi=600, bbox_inches='tight')
fig2.savefig('plots/q5_RFR_fig2.png', dpi=600, bbox_inches='tight')
fig3.savefig('plots/q5_RFR_fig3.png', dpi=600, bbox_inches='tight')

plt.show()
print('Criteria : variance')
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
