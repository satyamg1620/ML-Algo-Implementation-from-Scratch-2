# Answer 3

## Running time
### `fit()` time:
For the case of **real input discrete output**, the running time can be calculated,
$T\left(e, n, m\right) = \mathcal{O}(e(4n))$. $n$ is number of training datapoints and $e$ is number of estimators(Decision Trees of depth 1).  The 4n is contributed by  time complexity `fit()` function of Decision Tree of depth 1, `predict()` function of decision tree, `__calculate_error()` function and `__update_weights()` functions, each contibuting $\mathcal{O}(n)$ time complexity. The overall time complexity of `fit()` function is $T\left(e, n, m\right) = \mathcal{O}(en)$.

### `predict()` time

The time complexity is  $\mathcal{O}(en)$ where e is number of estimators(Decision Trees of depth 1) and n is number of test data points


|Description|Images|
|----|-------------|
|Decision Surface of stumps|<img src="plots/q3_adaboost_fig1.png" width=1000 alt="Decision Surface of stumps">|
|Decision Surface of AdaBoostClassifier|<img src="plots/q3_adaboost_fig2.png" width=400 alt="Decision Surface of AdaBoostClassifier">|
