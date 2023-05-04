import pandas as pd
import numpy as np


def entropy(Y: pd.Series, sample_weight: pd.Series = None) -> float:
    """
    Function to calculate the entropy
    """
    if sample_weight.__class__ == None.__class__:
        p = (Y.astype("object").value_counts() / Y.size)
    else:
        Y = pd.Series(Y, name='Y')
        W = pd.Series(sample_weight, name='W')
        df = pd.concat([Y, W], axis=1)
        p = df.groupby('Y').sum()['W']
        eps = 1e-7
    
    return -1 * (p * np.log2(p + eps)).sum()
    

def gini_index(Y: pd.Series, sample_weight: pd.Series=None) -> float:
    """
    Function to calculate the gini index
    """
    if sample_weight.__class__ == None.__class__:
        p = Y.value_counts() / Y.size
    else:
        Y = pd.Series(Y, name='Y')
        W = pd.Series(sample_weight, name='W')
        df = pd.concat([Y, W], axis=1)
        p = df.groupby('Y').sum()['W']
        
    return (p * (1 - p)).sum()


def information_gain(Y: pd.Series, attr: pd.Series, sample_weight: pd.Series=None) -> tuple:
    """
    Function to calculate the information gain. Not subtracting from entropy(Y)
    to improve speed. Does not affect the working.
    """
    assert Y.size == attr.size
    if attr.dtype == "category":
        counts = pd.crosstab(attr, Y).sum(axis=1)
        ratio = counts / counts.sum()
        p = pd.crosstab(attr, Y, normalize="index").replace(0, 1)
        reduction = (ratio * (-p * np.log2(p)).fillna(0).sum(axis=1)).sum()
        return -reduction, -1
    else:
        df = pd.DataFrame(np.array([attr, Y]).T, columns="attr,Y".split(","))
        df = df.sort_values(by="attr")
        uniq_idx = np.where(np.diff(df["attr"]) != 0)[0]
        if uniq_idx.size == 0:
            # cannot split about this feature
            return -np.inf, -1
        
        max_gain = -np.inf
        m_idx = uniq_idx[0]
        for idx in uniq_idx:
            up = df["Y"].iloc[: idx + 1]
            down = df["Y"].iloc[idx + 1 :]
            gain = - sum(sample_weight.iloc[:idx]) * entropy(up, sample_weight) - (sum(sample_weight.iloc[idx:])) * entropy(down, sample_weight)
            if gain > max_gain:
                max_gain = gain
                m_idx = idx
        return max_gain, m_idx

def gini_gain(Y: pd.Series, attr: pd.Series, sample_weight: pd.Series=None) -> tuple:
    """
    Function to calculate the gini gain. Not subtracting from gini_index(Y)
    to improve speed. Does not affect the working.
    """
    assert Y.size == attr.size
    if attr.dtype == "category":
        counts = pd.crosstab(attr, Y).sum(axis=1)
        ratio = counts / counts.sum()
        p = pd.crosstab(attr, Y, normalize="index")
        reduction = (ratio * (-p * (1 - p)).fillna(0).sum(axis=1)).sum()
        return -reduction, -1
    else:
        df = pd.DataFrame(np.array([attr, Y]).T, columns="attr,Y".split(","))
        df = df.sort_values(by="attr")
        uniq_idx = np.where(np.diff(df["attr"]) != 0)[0]
        if uniq_idx.size == 0:
            # cannot split about this feature
            return -np.inf, -1
        
        max_gain = -np.inf
        m_idx = uniq_idx[0]
        for idx in uniq_idx:
            up = df["Y"].iloc[: idx + 1]
            down = df["Y"].iloc[idx + 1 :]
            gain = - sum(sample_weight.iloc[:idx]) * gini_index(up, sample_weight) - (sum(sample_weight.iloc[idx:])) * gini_index(down, sample_weight)
            if gain > max_gain:
                max_gain = gain
                m_idx = idx
        return max_gain, m_idx


def variance_reduction(Y: pd.Series, attr: pd.Series, sample_weight: pd.Series = None) -> tuple:
    """
    Function to compute reduction in variance, or the split which gives maximum
    possible reduction in variance if the attribute is continuous. Not subtracting 
    from Y.var to improve speed. Does not affect the working.
    """
    assert Y.size == attr.size
    if attr.dtype == "category":
        classes = attr.unique()
        df = pd.DataFrame(np.array([attr, Y]).T, columns="attr,Y".split(","))
        weighted_variance = []
        for c in classes:
            vals = df.groupby("attr").get_group(c)["Y"]
            weight = vals.size / Y.size
            weighted_variance.append(weight * vals.var(ddof=0))
        return -sum(weighted_variance), -1
    else:
        df = pd.DataFrame(np.array([attr, Y]).T, columns="attr,Y".split(","))
        df = df.sort_values(by="attr")
        uniq_idx = np.where(np.diff(df["attr"]) != 0)[0]
        if uniq_idx.size == 0:
            # cannot split about this feature
            return -np.inf, -1
        
        max_gain = -np.inf
        m_idx = uniq_idx[0]
        for idx in uniq_idx:
            up = df["Y"].iloc[: idx + 1]
            down = df["Y"].iloc[idx + 1 :]
            gain = - ((idx / Y.size) * up.var(ddof=0) + ((Y.size - idx) / Y.size) * down.var(ddof=0))
            if gain > max_gain:
                max_gain = gain
                m_idx = idx
        return max_gain, m_idx
