# Optimized Random Forest for Health Risk Prediction (BRFSS 2015)

## Project Overview
This project builds and optimizes a **Random Forest classifier** to predict an individual’s overall health status using the **Behavioral Risk Factor Surveillance System (BRFSS) 2015** dataset.

Given the **highly imbalanced nature** of the data, the model is optimized using **ROC AUC** rather than accuracy. Hyperparameter tuning is performed using **RandomizedSearchCV** to identify the best-performing configuration.

---

## Problem Statement
Public health datasets often contain:
- High dimensionality
- Missing values
- Severe class imbalance

The objective is to accurately identify individuals with **poor health outcomes** while maintaining strong overall predictive performance.

---

## Dataset
- Source: BRFSS 2015 (CDC)
- Samples used: **100,000**
- Features: **313 numerical features**
- Target:
  - `1` → Healthy
  - `0` → Unhealthy

The dataset exhibits significant class imbalance, making ROC AUC, precision, and recall more appropriate evaluation metrics than accuracy.

---

## Data Preparation
- Selected numeric features only
- Removed high-missing or redundant health indicators
- Performed stratified train–test split (70% / 30%)
- Imputed missing values using training-set means
- Prevented data leakage by applying identical preprocessing to test data

---

## Model Architecture

### Random Forest Classifier
Random Forest is an ensemble learning method that:
- Combines multiple decision trees
- Uses bootstrap sampling (bagging)
- Randomly selects feature subsets at each split

This approach reduces variance and improves generalization compared to a single decision tree.

---

## Hyperparameter Optimization
To maximize performance, **RandomizedSearchCV** was used to explore multiple configurations across:
- Number of trees
- Tree depth
- Feature subset size
- Leaf node constraints
- Split thresholds
- Bootstrapping strategies

**Optimization metric:** ROC AUC  
**Cross-validation:** 3-fold  
**Search iterations:** 10

---

## Best Model Configuration
The optimized Random Forest achieved its best performance with:
- A deep but constrained tree structure
- A large ensemble size
- Partial feature sampling at each split
- Bootstrapped training samples

This configuration provided the best balance between bias and variance.

---

## Model Evaluation

### Performance Metrics
- **ROC AUC (Train):** ~0.87  
- **ROC AUC (Test):** ~0.87  
- **Precision (Test):** ~0.89  
- **Recall (Test):** ~0.95  

### Confusion Matrix (Test Set)
- Strong identification of healthy individuals
- Improved detection of unhealthy cases compared to baseline models
- Better precision–recall trade-off than both a single Decision Tree and a default Random Forest

---

## Key Insights
- Hyperparameter tuning significantly improves minority-class detection
- Optimized Random Forest reduces false negatives without excessive false positives
- Ensemble learning is well-suited for large, noisy public health datasets
- ROC AUC is a more reliable metric than accuracy for imbalanced problems

---

## Tools & Technologies
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## Future Improvements
- Introduce class-weighted learning to further reduce false positives
- Apply feature selection to reduce dimensionality
- Compare against Gradient Boosting and XGBoost
- Add explainability using SHAP or feature importance analysis

---

## Conclusion
This project demonstrates how **carefully tuned ensemble models** can deliver robust and interpretable predictions on large-scale, real-world health datasets—making them suitable for practical public health analytics and decision support.
