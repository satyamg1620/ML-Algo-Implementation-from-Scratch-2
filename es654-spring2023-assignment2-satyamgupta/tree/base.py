"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import information_gain, gini_gain, variance_reduction

np.random.seed(42)

@dataclass
class WeightedDecisionTree:
    criterion: Literal["information_gain", "gini_index"] = "information_gain"
    max_depth: int = None
    root: "Node" = None
    @dataclass
    class Node:
        oftype: Literal["categorical", "continuous", "leaf"] = "leaf"
        feature: str = ""
        label: str = ""
        boundary: Union[float, int] = 0.0
        value: Union[float, int, str] = 0.0
        children: list = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series = None) -> None:
        """
        Function to train and construct the decision tree
        """

        def classification(X: pd.DataFrame, y: pd.Series, depth: int, criterion: Literal, label: str, sample_weight: pd.Series = None) -> self.Node:
            if depth > 0:
                cNode = self.Node()
                cNode.label = label
                if X.empty:
                    cNode.value = y.value_counts().idxmax()
                    return cNode
                elif y.unique().size == 1:
                    cNode.value = y.iloc[0]
                    return cNode
                elif X.columns.size == 1 and X.iloc[:, 0].unique().size == 1:
                    # only one value for the only remaining feature
                    childNode = self.Node()
                    childNode.value = y.value_counts().idxmax()
                    childNode.feature = X.columns[0]
                    childNode.label = X.iloc[:, 0].unique()[0]
                    cNode.children.append(childNode)
                    return cNode
                else:
                    if criterion == "information_gain":
                        gain = X.apply(lambda x: information_gain(y, x, sample_weight), axis=0)
                    elif criterion == "gini_index":
                        gain = X.apply(lambda x: gini_gain(y, x, sample_weight), axis=0)
                    else:
                        raise ValueError("Invalid criterion!")
                    cNode.feature = gain.iloc[0].idxmax()

                    if gain[cNode.feature].iloc[1] == -1: # the best feature is categorical
                        cNode.oftype = "categorical"
                        G = X.groupby(cNode.feature).groups.items()
                        for val, idx in G:
                            X_new = X.loc[list(idx)].drop(
                                columns=[cNode.feature])
                            if not (X_new.empty or y[list(idx)].empty):
                                cNode.children.append(classification(X_new, y[list(idx)], depth - 1, criterion, val, sample_weight))
                        return cNode

                    else:
                        cNode.oftype = "continuous"
                        idx = int(gain.iloc[1][cNode.feature])

                        df = X.copy()
                        df["y"] = y
                        df = df.sort_values(by=cNode.feature)

                        cNode.boundary = df[cNode.feature].iloc[[idx, idx + 1]].mean()

                        X_left = df.iloc[: idx + 1].drop(columns=["y"])
                        y_left = df.iloc[: idx + 1]["y"]
                        X_right = df.iloc[idx + 1:].drop(columns=["y"])
                        y_right = df.iloc[idx + 1:]["y"]

                        cNode.children.append(classification(X_left, y_left, depth - 1, criterion, "True", sample_weight))
                        cNode.children.append(classification(X_right, y_right, depth - 1, criterion, "False", sample_weight))

                        return cNode
            else:
                cNode = self.Node()
                cNode.label = label
                cNode.value = y.value_counts().idxmax()
                return cNode

        def regression(X: pd.DataFrame, y: pd.Series, depth: int, label: str, sample_weight: pd.Series = None) -> self.Node:
            if depth > 0:
                cNode = self.Node()
                cNode.label = label
                if X.empty:
                    cNode.value = y.mean()
                    return cNode
                elif y.unique().size == 1:
                    cNode.value = y.iloc[0]
                    return cNode
                elif X.columns.size == 1 and X.iloc[:, 0].unique().size == 1:
                    # only one value for the only remaining feature
                    childNode = self.Node()
                    childNode.value = y.value_counts().idxmax()
                    childNode.feature = X.columns[0]
                    childNode.label = X.iloc[:, 0].unique()[0]
                    cNode.children.append(childNode)
                    return cNode
                else:
                    var_reduct = X.apply(lambda x: variance_reduction(y, x, sample_weight), axis=0)
                    cNode.feature = var_reduct.iloc[0].idxmax()

                    if var_reduct[cNode.feature].iloc[1] == -1:
                        cNode.oftype = "categorical"
                        G = X.groupby(cNode.feature).groups.items()
                        for val, idx in G:
                            X_new = X.loc[list(idx)].drop(
                                columns=[cNode.feature])
                            if not (X_new.empty or y[list(idx)].empty):
                                cNode.children.append(regression(X_new, y[list(idx)], depth - 1, val, sample_weight))
                        return cNode

                    else:
                        cNode.oftype = "continuous"
                        idx = int(var_reduct.iloc[1][cNode.feature])

                        df = X.copy()
                        df["y"] = y
                        df = df.sort_values(by=cNode.feature)

                        cNode.boundary = df[cNode.feature].iloc[[idx, idx + 1]].mean()

                        X_left = df.iloc[: idx + 1].drop(columns=["y"])
                        y_left = df.iloc[: idx + 1]["y"]
                        X_right = df.iloc[idx + 1:].drop(columns=["y"])
                        y_right = df.iloc[idx + 1:]["y"]

                        cNode.children.append(regression(X_left, y_left, depth - 1, "True", sample_weight))
                        cNode.children.append(regression(X_right, y_right, depth - 1, "False", sample_weight))

                        return cNode
            else:
                cNode = self.Node()
                cNode.label = label
                cNode.value = y.mean()
                return cNode

        if y.dtype == "category":
            if self.max_depth == None:
                self.root = classification(X, y, np.inf, self.criterion, "", sample_weight)
            else:
                self.root = classification(X, y, self.max_depth, self.criterion, "", sample_weight)
        else:
            if self.max_depth == None:
                self.root = regression(X, y, np.inf, "", sample_weight)
            else:
                self.root = regression(X, y, self.max_depth, "", sample_weight)

    def predict(self, X: pd.DataFrame, verbose: bool = False) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        def predict_datapoint(X: pd.Series, treeNode: self.Node, verbose=False) -> Union[float, int, str]:
            if verbose:
                print("predicting datapoint", X.to_list())
            while treeNode.oftype != "leaf":
                if treeNode.oftype == "categorical":
                    for i in treeNode.children:
                        if X[treeNode.feature] == i.label:
                            treeNode = i
                            break
                    else:
                        raise ValueError("Category not found!")
                elif treeNode.oftype == "continuous":
                    if X[treeNode.feature] <= treeNode.boundary:
                        treeNode = treeNode.children[0]
                    else:
                        treeNode = treeNode.children[1]
            return treeNode.value

        y_hat = X.apply(lambda x: predict_datapoint(x, self.root, verbose), axis=1)
        return y_hat

    def plot(self, tabwidth=4, increment=2) -> None:
        """
        Function to plot the tree
        tabwidth is the number of spaces in each level
        increment is the additional spaces after each level
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        def recurse(node, depth):
            if node.oftype == "leaf":
                print(depth * " " * tabwidth, end="")
                if node.value.__class__ == str:
                    print(f"{node.label} : {node.value}")
                else:
                    print(f"{node.label} : {node.value:,.2f}")
            elif node.oftype == "categorical":
                print(depth * " " * tabwidth, end="")
                if depth == 0:
                    print(f"{node.feature} :")
                else:
                    print(f"{node.label} : {node.feature}")
                for child in node.children:
                    recurse(child, depth + increment)
            elif node.oftype == "continuous":
                print(depth * " " * tabwidth, end="")
                if depth == 0:
                    print(f"{node.feature} <= {node.boundary:,.2f} ?")
                else:
                    print(f"{node.label}: {node.feature} <= {node.boundary:,.2f} ?")
                for child in node.children:
                    recurse(child, depth + increment)

        recurse(self.root, 0)
        
    def plot_Decision_Boundary(self, X: pd.DataFrame, y: pd.Series, res:int = 100):
        """
        Plots decision boundaries and the given datapoints as scatter plot over it.
        """
        
        # define bounds of the domain
        min1, max1 = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        min2, max2 = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
        x1grid = np.linspace(min1, max1, res)
        x2grid = np.linspace(min2, max2, res)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1, r2))
        
        plt.figure()
        y_hat = self.predict(pd.DataFrame(grid))
        z_z = y_hat.to_numpy().reshape(xx.shape)
        plt.contourf(xx, yy, z_z, cmap='Paired')

        for class_value in range(2):
            row_ix = np.where(y == class_value)
            plt.scatter(X.to_numpy()[row_ix,0], X.to_numpy()[row_ix,1],  cmap='Paired')
            
