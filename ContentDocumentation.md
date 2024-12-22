## Measuring Performance in Linear Regression Models

**Measures of Error**

![image](https://github.com/user-attachments/assets/7c30c60f-bde0-4516-8c83-b6734f912b1f)

- Mean Squared Error (MSE)

**Theoretical Explanation:** 
   - MSE measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual values. It is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.
   - MSE is a robust measure of accuracy that punishes larger errors more significantly, making it useful when big mistakes are particularly undesirable.

![image](https://github.com/user-attachments/assets/0c6ea079-6314-4d42-83e9-00979e0b749e)

- Root Mean Squared Error (RMSE)

**Theoretical Explanation:** 
   - RMSE is the square root of the mean of the square of all of the error. It is a measure that is used to quantify the difference between values predicted by a model and the values observed.
   - RMSE is the square root of the average of the square of all of the errors. It provides an estimate of the magnitude of the error. By taking the square root of MSE, RMSE lowers the sensitivity to outliers compared to MSE and is on the same scale as the data, making interpretation straightforward.

![image](https://github.com/user-attachments/assets/7c06ed45-5414-4aae-8037-92de9458673f)

- Mean Absolute Error (MAE)

**Theoretical Explanation:** 
- MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

![image](https://github.com/user-attachments/assets/b7893d9b-548a-4472-ab3a-4a1332c58d70)

- Mean Absolute Percentage Error (MAPE)

**Theoretical Explanation:**
   - MAPE measures the size of the error in percentage terms. It is calculated as the average of the unsigned percentage error.
   - It is intuitive for understanding the relative magnitude of the prediction errors.

![image](https://github.com/user-attachments/assets/cecbe50b-e172-41f7-af1d-8dd3b8d0efc0)

![image](https://github.com/user-attachments/assets/5b487988-3a4b-40a9-b459-05510c91a32a)

**Measure of Explained Variance**
- R² (Coefficient of Determination)

**Theoretical Explanation:**
   - R-squared is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. It indicates how well data fit a statistical model – more specifically, a line. R-squared values range from 0 to 1 and are often expressed as percentages. A higher R-squared percentage reflects a model that explains a higher proportion of the variance.

![image](https://github.com/user-attachments/assets/93f580d8-d50d-4868-b2cd-08587e6c4ae6)

- Adjusted R²

**Theoretical Explanation:**
   - Adjusted R-squared also indicates how well terms fit a curve or line but adjusts for the number of terms in a model. Unlike R-squared, adjusted R-squared increases only if a new term improves the model more than would be expected by chance. It accounts for the number of predictors in the model and can help to prevent overfitting by penalizing excessive use of non-contributing predictors.

![image](https://github.com/user-attachments/assets/76f4ca6d-e5b0-4f2e-a6a5-e830f65035c0)

![image](https://github.com/user-attachments/assets/d778e84b-f99d-435f-bd8e-9ca058ba2cc5)

**Model Selection Criteria**
- Akaike Information Criterion (AIC)

**Theoretical Explanation:** 
   - AIC is a criterion for model selection among a finite set of models. It is founded on information theory: it offers an estimate of the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model. AIC deals with the trade-off between the goodness of fit of the model and the complexity of the model.

![image](https://github.com/user-attachments/assets/c9aafb40-6cf7-4eed-bb3e-ff160f960a56)

- Bayesian Information Criterion (BIC)

***Theoretical Explanation:**
   - BIC is similar to AIC but with a stronger penalty for including additional variables in the model. It is also based on information theory, but it adds a penalty term for the number of parameters in the model. This makes BIC more stringent about including unnecessary variables, thus it tends to penalize complex models more than AIC.

![image](https://github.com/user-attachments/assets/8aaa9603-2ac4-4342-8ed1-6ff0af799323)

## Measuring Performance in Classification Models
**Class-probability based metrics**
* Log-likelihood: the probability of observing the given data under a specific model. In the context of classification, it assesses how probable the observed class labels are, given the predicted probabilities from the model.

![image](https://github.com/user-attachments/assets/4dfcc4af-1cd1-47bd-96e6-03e8ac10558d)

* Gini: a measure derived from the Lorenz curve, often used to quantify inequality or concentration. In classification, it measures the inequality among values of a frequency distribution (such as levels of income). For model performance, it reflects the discriminatory power of the model.

![image](https://github.com/user-attachments/assets/9a675136-f1c0-4269-bf73-26b44563079f)

* Entropy:
- Entropy in the context of classification is a measure of impurity or disorder. It quantifies the amount of uncertainty involved in predicting the class labels of a dataset.
- Entropy in information theory, often used in the field of machine learning, particularly in decision tree algorithms, is a measure of the unpredictability or the randomness of a data set. 
- In the context of classification, entropy is used to quantify the impurity or disorder within a set of elements, which in this case are the class labels of the data points.

![image](https://github.com/user-attachments/assets/0ec63370-2344-4ed3-967a-e4693d36861a)

**Confusion Matrix**

![image](https://github.com/user-attachments/assets/32a27ed1-2dc4-4690-ad32-608552c0ad05)

Source: ENCORD https://encord.com/glossary/confusion-matrix/

* Accuracy

![image](https://github.com/user-attachments/assets/d39ed0e8-9022-4427-a964-7ebcbac1c4a3)

* Misclassification Rate

![image](https://github.com/user-attachments/assets/3480206c-4b4e-4361-bdd2-9510f0d95991)

* Sensitivity

![image](https://github.com/user-attachments/assets/c2f98576-9afd-47e7-926e-0d8fefff36bb)

* Specificity

![image](https://github.com/user-attachments/assets/a32df2fd-98ec-4555-a341-6714855fcea4)

* Precision

![image](https://github.com/user-attachments/assets/27c12c0d-c861-4dec-b50c-7dfa1496815c)

**Type 1 and Type 2 Errors**
**Type I Error (False Positive)**
- Definition: A Type I error occurs when the model incorrectly predicts the positive class, also known as a "false positive." For example, a medical test that incorrectly diagnoses a healthy patient as sick commits a Type I error.

- Implications: Type I errors are critical in fields like medicine or judicial systems where the consequences of falsely identifying an issue (like a disease or crime) can be severe.

**Type II Error (False Negative)**
- Definition: A Type II error occurs when the model fails to detect the positive class, also known as a "false negative." For example, a diagnostic test that fails to detect a condition that is present commits a Type II error.

- Implications: Type II errors are particularly problematic in scenarios like cancer screening or fraud detection, where failing to detect an existing problem can have dire consequences.

**Contextual Importance:**
* ***Sensitivity is directly related to Type II errors***, as it measures the ability to correctly identify positives (thus avoiding false negatives).
* ***Specificity is related to Type I errors***, as it measures the ability to correctly identify negatives (thus avoiding false positives).

![image](https://github.com/user-attachments/assets/1e88d569-d5ea-4d1f-8bb2-efb1af3bd401)

**Area under the ROC curve (AUC)**
The Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve is a performance measurement for classification problems at various threshold settings. The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. It plots the True Positive Rate (TPR, Sensitivity) against the False Positive Rate (FPR, 1 - Specificity) at different threshold levels.

**Calculation:**
- The AUC measures the entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1). This metric provides an aggregate measure of performance across all possible classification thresholds. The value of the AUC ranges from 0 to 1, where:
    * 0.5 denotes a model with no discriminative ability (equivalent to random guessing).
    * 1 denotes a perfect model that achieves 100% sensitivity (no false negatives) and 100% specificity (no false positives).

**Threshold in Classification Models**
- In binary classification tasks, a model typically outputs a probability or a score indicating the likelihood that a given instance belongs to the positive class. The threshold is a value that determines the point at which a probability score is classified as positive or negative. For example, in a binary classification for disease diagnosis:
    * Probability Score: A logistic regression model might predict a probability of 0.7 that a patient has a particular disease.
    * Threshold: If the threshold is set at 0.5, any probability greater than or equal to 0.5 is classified as 'disease' (positive class), and below 0.5 as 'no disease' (negative class).

**Role of Threshold in ROC Curves**
- The ROC curve illustrates the performance of a classifier without committing to a specific threshold, by showing the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR) at various threshold settings:
    * True Positive Rate (Sensitivity): The proportion of actual positives correctly identified as such (e.g., the proportion of sick patients correctly diagnosed as sick).
    * False Positive Rate: The proportion of negative instances incorrectly classified as positive (e.g., the proportion of healthy patients incorrectly diagnosed as sick).

**Changing Threshold Levels**
* Increasing the Threshold: Higher thresholds mean the classifier must be more certain before classifying a positive. This typically reduces both false positives and true positives, leading to a lower FPR but also potentially a lower TPR.
* Decreasing the Threshold: Lower thresholds result in more instances being classified as positive. This increases sensitivity (TPR) because more actual positives are caught, but also increases the FPR because more negatives are falsely classified as positives.

![image](https://github.com/user-attachments/assets/404c1432-20f7-4007-973a-fac46de7138d)

**Choosing the Best Threshold**
To choose the best threshold from the ROC curve, consider the following approaches:
1. Maximize TPR while minimizing FPR: Look for a point on the curve that is closest to the top-left corner of the plot (near perfect sensitivity and specificity). This point represents a good balance between sensitivity (TPR) and specificity (1 - FPR), minimizing both false negatives and false positives.

2. Youden’s Index: This method selects the threshold that maximizes the difference between the True Positive Rate and the False Positive Rate. Mathematically, it is calculated as: 
    * J=Sensitivity+Specificity−1 This statistic will help you to find the threshold where the sum of sensitivity and specificity is maximized, which corresponds to the point on the ROC curve that is furthest north-west.

3. Cost-based approach: If there are different costs associated with false positives and false negatives, you can assign costs to them and choose a threshold that minimizes the total cost.

4. Precision-Recall Trade-off: In some cases, especially with imbalanced datasets, you might prefer to look at the precision-recall curve instead and choose a threshold that balances these two metrics according to your business needs.

