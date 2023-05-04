# Answer 4

We see that adding more processes has the biggest effect when n_estimators is large. When n_estimators is small, the overhead of setting up the parallel computation (creating multiprocessing pool and distributing the data to the processes) dominates. Prediction time in fact suffers as we add more processes for small n_estimators. We need to plot for a really huge number of n_estimators to see any benefit of parallel computation for prediction. The time complexity of fit is $\mathcal{O}(EMN^2\log N)$ and predict is $\mathcal{O}(EN\log N)$ for $E$ estimators.

|Description|Images|
|----|-------------|
|Decision surface for BaggingClassifier for each estimator(iteration)|<img src="plots/q4_bagging_fig1.png" width=1000 alt="Decision surface for BaggingClassifier for each estimator(iteration)">|
|Decision Surface of BaggingClassifier|<img src="plots/q4_bagging_fig2.png" width=400 alt="Decision Surface of BaggingClassifier">|
|Mean accuracy (and std) vs n_estimators (20 iterations)|<img src="plots/q4_accuracy.png" width=400 alt="Decision Surface of BaggingClassifier">|
|Running time of fit for different n_jobs and n_estimators|<img src="plots/q4_fit_times.png" width=400 alt="Running time of fit for different n_jobs and n_estimators">|
|Running time of predict for different n_jobs and n_estimators|<img src="plots/q4_predict_times.png" width=400 alt="Running time of fit for different n_jobs and n_estimators">|

