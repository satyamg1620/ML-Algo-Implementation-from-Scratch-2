import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import copy


class AdaBoostClassifier():
    def __init__(self, base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=3): # Optional Arguments: Type of estimator
        """
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.stumps = []
        self.train_error = []
        self.alpha_list = []
        self.wts = []
        self.X_train = None
        self.y_train = None
        
        pass
    
    def __calculate_error(self, y, y_pred, w):
        err = sum(w[np.not_equal(y, y_pred)]) / sum(w)
        return err
    
    def __update_weights(self, alpha, w, y, y_pred):
        w = w * np.exp(alpha * (np.not_equal(y, y_pred)))
        return w / sum(w)
    
    def __calculate_alpha(self, err):
        """
        Following Algorithm 2 SAMME from the paper "Multi-class AdaBoost" by
        Zhu et al.
        """
        eps = 1e-7
        return np.log((1 - err-eps) / (err + eps)) + np.log(self.n_classes - 1)

    def fit(self, X:pd.DataFrame, y:pd.Series):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        
        self.n_classes = len(y.unique())
        
        w = np.ones(len(y)) / len(y)
        
        #iterate
        for i in range(self.n_estimators):
                
            base_clf = copy.deepcopy(self.base_estimator)
            base_clf.fit(X, y, sample_weight=w)
            y_pred = base_clf.predict(X)
            
            self.stumps.append(base_clf)
            self.wts.append(w.copy().tolist())
            
            err = self.__calculate_error(y, y_pred, w)
            
            if err > 1 / self.n_classes:
                break
            
            self.train_error.append(err)
            
            alpha = self.__calculate_alpha(err)
            self.alpha_list.append(alpha)
            
            w = self.__update_weights(alpha, w, y, y_pred)
            
            print(f"{i}th iteration; error: {err}")
        
        self.n_estimators = i + 1
        
        return True

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        
        n_samples, _ = X.shape
        y_pred_multiclass = pd.DataFrame(0, index=range(n_samples),columns=range(self.n_classes))
        for alpha, estimator in zip(self.alpha_list, self.stumps):
            X_est = estimator.predict(X)
            for i in range(len(X_est)):
                y_pred_multiclass.iloc[i, X_est[i]] += alpha
        
        y_pred = y_pred_multiclass.idxmax(axis=1)
        return pd.Series(y_pred)
        

    def plot(self, X, y):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        
        
        # define bounds of domain for plotting decision boundaries
        min1, max1 = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        min2, max2 = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
        
        x1grid = np.arange(min1, max1, 0.1)
        x2grid = np.arange(min2, max2, 0.1)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1,r2))
        
        fig1, ax = plt.subplots(1, self.n_estimators, figsize=(3*self.n_estimators, 3))
        for i in range(self.n_estimators):
            ax[i].set_title(f'alpha={round(self.alpha_list[i],2)}')
            y_hat = self.stumps[i].predict(grid)
            zz = y_hat.reshape(xx.shape)
            ax[i].contourf(xx, yy, zz, cmap='Paired')
            for class_value in range(2):
                row_ix = list(np.where(y.astype(int) == class_value))
                w = np.array(self.wts[i])[row_ix]
                ax[i].scatter(X.to_numpy()[row_ix,0], X.to_numpy()[row_ix,1], s=w*400)
            ax[i].axis('square')
        plt.tight_layout()
        
        plt.figure(2)
        y_hat = self.predict(grid.copy())
        zz = y_hat.to_numpy().reshape(xx.shape)
        plt.contourf(xx, yy, zz, cmap='Paired')
        plt.title(label="Decision Surface of AdaBoostClassifier")
        for class_value in range(2):
            row_ix = np.where(y == class_value)
            plt.scatter(X.to_numpy()[row_ix,0], X.to_numpy()[row_ix,1],  cmap='Paired')
        
        return [fig1, plt.figure(2)]
