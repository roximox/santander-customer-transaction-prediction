# Pre-Final Model Comparison Portfolio

| Model                                      | Member    | ROC-AUC mean ± std   | Average Precision mean ± std   |       F1 |   Precision |   Recall |   Balanced Accuracy |   Train-validation gap |   Fit time |
|:-------------------------------------------|:----------|:---------------------|:-------------------------------|---------:|------------:|---------:|--------------------:|-----------------------:|-----------:|
| Logistic Regression L2 Baseline            | Member 01 | 0.859188 ± 0.003239  | 0.507566 ± 0.008490            | 0.390361 |    0.688813 | 0.272484 |            0.629343 |             0.00233683 |   0.644985 |
| Logistic Regression L2 Balanced            | Member 01 | 0.859011 ± 0.003121  | 0.506430 ± 0.008519            | 0.416059 |    0.28456  | 0.773541 |            0.778128 |             0.00281155 |   0.66912  |
| Logistic Regression L2 C=0.01 (unweighted) | Member 01 | 0.859201 ± 0.003237  | 0.507592 ± 0.008491            | 0.385881 |    0.691427 | 0.267758 |            0.627181 |             0.002325   |   6.6038   |
| Logistic Regression L2 C=0.01 (balanced)   | Member 01 | 0.859017 ± 0.003122  | 0.506454 ± 0.008518            | 0.416119 |    0.284599 | 0.773666 |            0.778194 |             0.00280527 |  12.434    |

CV results only; final-test evaluation has not been performed.
