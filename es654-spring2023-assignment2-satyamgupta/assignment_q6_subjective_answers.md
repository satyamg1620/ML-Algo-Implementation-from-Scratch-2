# Answer 6

For small datasets, we usually fix the number of leafs to be 4, or max_depth to be 2. The process is as follows:
* Fit an estimator on the given dataset
* Predict using the ensemble (initially a single estimator) on the test set and compute the difference from the target (these are the residuals)
* Fit another estimator on the residuals and add it to the ensemble with its contribution scaled by the learning rate
* Repeat the process till desired accuracy is reached, or stop when it is infeasible to add more estimators.