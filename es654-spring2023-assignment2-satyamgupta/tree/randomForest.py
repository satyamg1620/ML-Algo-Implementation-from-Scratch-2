from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        
        df = pd.concat([X, y], axis=1)
        _, n_features = X.shape
        for i in range(self.n_estimators):
            bootstrap = df.sample(frac=1, replace=True)
            tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, max_features=int(n_features**0.5))
            tree.fit(bootstrap.iloc[:,:-1], bootstrap.iloc[:,-1])
            self.trees.append(tree)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = []
        for ctree in self.trees:
            predictions.append(pd.Series(ctree.predict(X)))
        return pd.concat(predictions, axis=1).mode(axis=1).iloc[:,0].squeeze()

    def plot(self, X, y, res=100):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        
        
        
        fig1, ax = plt.subplots(1, self.n_estimators, figsize=(3*self.n_estimators, 3))
        for i in range(self.n_estimators):
            plot_tree(self.trees[i], ax=ax[i])
        plt.tight_layout()
        
        # define bounds of domain for plotting decision boundaries
        min1, max1 = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        min2, max2 = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
        
        # make grid
        x1grid = np.linspace(min1, max1, res)
        x2grid = np.linspace(min2, max2, res)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        
        _, n_features = X.shape
        X_projected = np.zeros((res**2, n_features))
        
        cmap_list_l = [plt.cm.tab20(2*i + 1) for i in range(n_features)]
        cmap_list_d = [plt.cm.tab20(2*i) for i in range(n_features)]
        cmap = ListedColormap(cmap_list_l)

        for i in range(n_features):
            X_projected[:, i] = X.iloc[:, i].mean()
        X_projected[:, 0], X_projected[:, 1] = np.squeeze(r1), np.squeeze(r2)
        
        fig2, ax = plt.subplots(1, self.n_estimators, figsize=(3*self.n_estimators, 3))
        for i in range(self.n_estimators):
            
            y_hat = self.trees[i].predict(X_projected)
            zz = y_hat.reshape(xx.shape)
            
            ax[i].contourf(xx, yy, zz, cmap=cmap)
            ax[i].set_title(f'Tree #{i+1}')
            for class_value in range(len(y.unique())):
                row_ix = list(np.where(y.astype(int) == class_value))
                ax[i].scatter(X.to_numpy()[row_ix,0], X.to_numpy()[row_ix,1], c=[cmap_list_d[class_value]])
        plt.tight_layout()
            
        fig3, ax = plt.subplots()
        
        y_hat = self.predict(X_projected)
        zz = y_hat.to_numpy().reshape(xx.shape)
        ax.contourf(xx, yy, zz, cmap=cmap)
        
        ax.set_title(label="Decision Surface of RandomForest")
        
        for class_value in range(len(y.unique())):
            row_ix = np.where(y == class_value)
            ax.scatter(X.to_numpy()[row_ix,0], X.to_numpy()[row_ix,1], c=[cmap_list_d[class_value]])
        
        return [fig1, fig2, fig3]


class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='squared_error', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        
        df = pd.concat([X, y], axis=1)
        _, n_features = X.shape
        for i in range(self.n_estimators):
            bootstrap = df.sample(frac=1, replace=True)
            tree = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth, max_features=int(n_features**0.5))
            tree.fit(bootstrap.iloc[:,:-1], bootstrap.iloc[:,-1])
            self.trees.append(tree)

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = []
        for rtree in self.trees:
            predictions.append(pd.Series(rtree.predict(X)))
        return pd.concat(predictions, axis=1).mean(axis=1).squeeze()

    def plot(self, X, y, res=100):
        """
        Function to plot for the RandomForestRegressor.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        
        fig1, ax = plt.subplots(1, self.n_estimators, figsize=(3*self.n_estimators, 3))
        for i in range(self.n_estimators):
            plot_tree(self.trees[i], ax=ax[i])
        plt.tight_layout()
        
        # define bounds of domain for plotting decision boundaries
        min1, max1 = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        min2, max2 = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
        
        # make grid
        x1grid = np.linspace(min1, max1, res)
        x2grid = np.linspace(min2, max2, res)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        
        _, n_features = X.shape
        X_projected = np.zeros((res**2, n_features))
        
        cmap_list_l = [plt.cm.Paired(2*i) for i in range(n_features)]
        cmap_list_d = [plt.cm.Paired(2*i + 1) for i in range(n_features)]
        cmap = ListedColormap(cmap_list_l)

        for i in range(n_features):
            X_projected[:, i] = X.iloc[:, i].mean()
        X_projected[:, 0], X_projected[:, 1] = np.squeeze(r1), np.squeeze(r2)
        
        fig2, ax = plt.subplots(1, self.n_estimators, figsize=(3*self.n_estimators, 3))
        for i in range(self.n_estimators):
            y_hat = self.trees[i].predict(X_projected)
            zz = y_hat.reshape(xx.shape)
            im = ax[i].contourf(xx, yy, zz, cmap='jet')
            ax[i].set_title(f'Tree #{i+1}')                
        fig2.colorbar(im, ax=ax[-1])
        plt.tight_layout()
        
        fig3, ax = plt.subplots()
        
        y_hat = self.predict(X_projected)
        zz = y_hat.to_numpy().reshape(xx.shape)
        im = ax.contourf(xx, yy, zz, cmap='jet')
        fig3.colorbar(im, ax=ax)
        ax.set_title(label="Decision Surface of RandomForest")
        
        return [fig1, fig2, fig3]