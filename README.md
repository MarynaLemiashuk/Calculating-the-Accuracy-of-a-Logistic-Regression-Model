# 📊 Calculating the Accuracy of a Logistic Regression Model

This repository contains a Jupyter Notebook that demonstrates how to **build**, **interpret**, and **evaluate** a logistic regression model using Python.  
The notebook covers the full workflow: **data preparation**, **model fitting**, **interpretation**, **prediction**, **evaluation**, and **visualization**.  

---

## 🚀 Project Overview
- Load and clean **Bank Marketing dataset**  
- Perform **logistic regression** with `statsmodels`  
- Compare **single-variable** and **multi-variable** models  
- Evaluate performance using a **confusion matrix** and **accuracy score**  
- Visualize results for better interpretation  

---

## 🛠 Tech Stack
- Python  
- **Pandas / NumPy** → data handling  
- **Statsmodels** → logistic regression modeling  
- **Matplotlib / Seaborn** → data visualization  
- **SciPy** → statistical functions

  # 📊 Key Insights from Logistic Regression Analysis

This repository contains a Jupyter Notebook that demonstrates how to **analyze**, **interpret**, and **evaluate** a logistic regression model using Python. The notebook covers the full workflow: **data cleaning**, **model fitting**, **coefficient interpretation**, **prediction**, **evaluation**, and **visualization**.  

---

## 🚀 Key Insights
- **Duration** is a strong predictor of whether a client subscribes.  
- **Odds ratios** close to 1 indicate small practical effects despite statistical significance.  
- **Logistic regression** helps understand probabilities and odds in classification problems.  

---

## 🎯 What You’ll Learn
- How to **clean and prepare data** for logistic regression  
- How to **interpret coefficients and p-values**  
- How to **evaluate performance** with a confusion matrix and accuracy  
- How to **visualize binary classification outcomes**  

---

## 📂 Workflow

### 1. Data Preparation
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

raw_data = pd.read_csv("Bank_data.csv")
data = raw_data.copy()
data = data.drop(['Unnamed: 0'], axis=1)
data['y'] = data['y'].map({'yes':1,'no':0})

## 2. Simple Logistic Regression
```python
y = data['y']
x1 = data['duration']

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
results_log.summary()

## 3. Multiple Logistic Regression
```python
estimators = ['interest_rate','march','credit','previous','duration']
X1 = data[estimators]
y = data['y']

X = sm.add_constant(X1)
reg_logit = sm.Logit(y, X)
results_logit = reg_logit.fit()
results_logit.summary2()

## 4. Confusion Matrix & Accuracy
```python
def confusion_matrix(data, actual_values, model):
    pred_values = model.predict(data)
    bins = np.array([0, 0.5, 1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    return cm, accuracy

confusion_matrix(X, y, results_logit)


