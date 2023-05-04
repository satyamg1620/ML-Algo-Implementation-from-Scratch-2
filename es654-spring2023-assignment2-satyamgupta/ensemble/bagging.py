import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import multiprocessing
import numpy as np
from time import process_time_ns


class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100, n_jobs=1):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        :param n_jobs: The number of parallel processes
        '''
        
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []
        
        # nepj = number of estimators per job
        if n_jobs > n_estimators:
            self.n_jobs = n_estimators
            self.nepj = 1
            self.remainder = 0
        elif n_estimators % n_jobs == 0:
            self.n_jobs = n_jobs
            self.nepj = n_estimators // n_jobs
            self.remainder = 0
        else:
            self.n_jobs = n_jobs + 1
            self.nepj = n_estimators // n_jobs
            self.remainder = n_estimators % n_jobs

    def __get_estimators(self, df, n_estimators):
        """
        Returns a list of n_estimators using Bagging
        """
        models = []
        for i in range(n_estimators):
            # random seed depends on time
            bootstrap = df.sample(frac=1, replace=True, random_state=process_time_ns()%100000)
            model = deepcopy(self.base_estimator)
            model.fit(bootstrap.iloc[:,:-1], bootstrap.iloc[:,-1])
            models.append(model)
        return models

    def __fit_process(self, df, i):
        """
        Each process will call this function to fit their share of models
        """
        if self.remainder != 0 and i == self.n_jobs - 1:
            n_iterations = self.remainder
        else:
            n_iterations = self.nepj

        return self.__get_estimators(df, n_iterations)

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        
        # define and store bounds of domain for plotting decision boundaries
        min1, max1 = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        min2, max2 = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
        self.bounds = [[min1, max1], [min2, max2]]
        
        self.models = []
        df = pd.concat([X, y], axis=1)
        
        if self.n_jobs == 1:
            # don't bother creating the pool, unnecessary overhead
            self.models = self.__get_estimators(df, self.n_estimators)
        else:
            # create a pool of worker processes
            pool = multiprocessing.Pool(processes=self.n_jobs)
    
            results = [pool.apply_async(self.__fit_process, args=(df,i)) for i in range(self.n_jobs)]
            [self.models.extend(r.get()) for r in results]
    
            # close the pool and wait for the worker processes to terminate
            pool.close()
            pool.join()
    
    def __get_predictions(self, X, lo, hi):
        """
        Returns a list of predictions from estimators with index going lo to hi 
        """
        predictions = []
        for model in self.models[lo:hi]:
            predictions.append(model.predict(X))
        return predictions
        
    def __predict_process(self, X, i):
        """
        Each process will call this function to predict from their share of models
        """
        if self.remainder != 0 and i == self.n_jobs - 1:
            n_iterations = self.remainder
        else:
            n_iterations = self.nepj
        return self.__get_predictions(X, i*self.nepj, i*self.nepj + n_iterations)
    
    def __majority(self, predictions):
        return pd.concat(predictions, axis=1).mode(axis=1).iloc[:,0].squeeze()
    
    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        """

        if self.n_jobs == 1:
            # don't bother creating the pool, unnecessary overhead
            predictions = self.__get_predictions(X, 0, self.n_estimators)
            return self.__majority(predictions)

        # create a pool of worker processes
        pool = multiprocessing.Pool(processes=self.n_jobs)
        
        results = [pool.apply_async(self.__predict_process,
                                    args=(X, i)) for i in range(self.n_jobs)]
        
        predictions = []
        [predictions.extend(r.get()) for r in results]
        return self._majority(predictions)

    def plot(self, X, y):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
                
        # define bounds of domain for plotting decision boundaries
        min1, max1 = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        min2, max2 = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
        
        # make grid
        x1grid = np.arange(min1, max1, 0.1)
        x2grid = np.arange(min2, max2, 0.1)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1,r2))
        
        fig1, ax = plt.subplots(1, self.n_estimators, figsize=(3*self.n_estimators, 3))
        for i in range(self.n_estimators):
            
            y_hat = self.models[i].predict(grid.copy())
            zz = y_hat.to_numpy().reshape(xx.shape)
            
            ax[i].contourf(xx, yy, zz, cmap='Paired')
            ax[i].set_title(f'Round #{i+1}')
            for class_value in range(len(y.unique())):
                row_ix = list(np.where(y.astype(int) == class_value))
                ax[i].scatter(X.to_numpy()[row_ix,0], X.to_numpy()[row_ix,1])

        plt.tight_layout()
        
        plt.figure(2)
        y_hat = self.predict(grid)
        zz = y_hat.to_numpy().reshape(xx.shape)
        plt.contourf(xx, yy, zz, cmap='Paired')
        plt.title(label="Decision Surface of BaggingClassifier")
        
        for class_value in range(2):
            row_ix = np.where(y == class_value)
            plt.scatter(X.to_numpy()[row_ix,0], X.to_numpy()[row_ix,1],  cmap='Paired')
        
        return [plt.figure(1), plt.figure(2)]