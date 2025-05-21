Below is a detailed analysis of the code provided in the Jupyter Notebook file `Assignment02(Logistic R Clasfition).ipynb`. This notebook implements a logistic regression model to predict breast cancer survival status based on a dataset stored in "Breast_Cancer.csv". The code follows a standard machine learning workflow: loading data, exploring and preprocessing it, training a model, and evaluating its performance. I’ll break it down step by step.

---

## **1. Importing Libraries**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
```
- **Purpose**: Imports the necessary Python libraries for the analysis.
  - `pandas`: Handles data manipulation and loads the dataset into a DataFrame.
  - `numpy`: Supports numerical operations (though minimally used here).
  - `sklearn.model_selection`: Provides `train_test_split` to split data into training and testing sets.
  - `sklearn.linear_model`: Supplies the `LogisticRegression` class for modeling.
  - `sklearn.metrics`: Offers tools like `accuracy_score`, `confusion_matrix`, and `classification_report` for evaluation.
  - `sklearn.preprocessing`: Includes `LabelEncoder` for converting categorical data into numerical format.
- **What’s Happening**: Sets up the foundation for data processing and machine learning tasks.

---

## **2. Loading the Dataset**
```python
data = pd.read_csv("Breast_Cancer.csv")
```
- **Purpose**: Loads the breast cancer dataset from the CSV file into a pandas DataFrame named `data`.
- **What’s Happening**: The dataset contains patient-related features (e.g., age, race, tumor size) and a target variable (`Status`), which indicates survival outcome ("Alive" or "Dead").

---

## **3. Exploratory Data Analysis (EDA)**
The code includes several steps to explore the dataset:
```python
data.head()
data.info()
data.isna().sum()
data.duplicated().sum()
data.shape
data.columns
```
- **Purpose**: Understands the dataset’s structure and quality.
  - `data.head()`: Displays the first five rows, showing columns like `Age`, `Race`, `T Stage`, and `Status`.
  - `data.info()`: Lists column names, data types (e.g., `int64`, `object`), and confirms no missing values (4024 non-null entries per column initially).
  - `data.isna().sum()`: Verifies no missing values (all zeros).
  - `data.duplicated().sum()`: Checks for duplicate rows (one duplicate found, as `shape` changes later).
  - `data.shape`: Returns (4024, 16) initially, indicating 4024 rows and 16 columns.
  - `data.columns`: Lists all 16 column names, e.g., `Age`, `Marital Status`, `Survival Months`.
- **What’s Happening**: Provides a snapshot of the data, confirming it’s clean except for one duplicate row.

---

## **4. Removing Duplicates**
```python
data.drop_duplicates(inplace=True)
```
- **Purpose**: Eliminates duplicate rows to ensure data integrity.
- **What’s Happening**: Reduces the dataset from 4024 to 4023 rows (confirmed by `data.shape` output: `(4023, 16)`), removing one redundant entry.

---

## **5. Mapping the Target Variable**
```python
status_mapping = {'Alive': 1, 'Dead': 0}
data['Status'] = data['Status'].map(status_mapping)
```
- **Purpose**: Converts the categorical `Status` column into numerical values.
- **What’s Happening**: 
  - Original values: "Alive" and "Dead".
  - Mapped values: "Alive" → 1, "Dead" → 0.
  - Verified by `data["Status"].unique()`, showing `[1, 0]`. This is necessary for logistic regression, which requires a numerical target.

---

## **6. Encoding Categorical Variables**
The code encodes multiple categorical columns into numerical values:
```python
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

data['Race'] = data['Race'].map({'White': 0, 'Black': 1, 'Other': 2})
data['Marital Status'] = LE.fit_transform(data['Marital Status'])
data['T Stage '] = LE.fit_transform(data['T Stage '])
data['N Stage'] = LE.fit_transform(data['N Stage'])
data['6th Stage'] = LE.fit_transform(data['6th Stage'])
data['differentiate'] = LE.fit_transform(data['differentiate'])
data['Grade'] = LE.fit_transform(data['Grade'].replace(' anaplastic; Grade IV', '4'))
data['A Stage'] = LE.fit_transform(data['A Stage'])
data['Estrogen Status'] = LE.fit_transform(data['Estrogen Status'])
data['Progesterone Status'] = LE.fit_transform(data['Progesterone Status'])
```
- **Purpose**: Transforms categorical features into numerical format for modeling.
- **What’s Happening**:
  - **Manual Mapping**: `Race` is mapped explicitly (`White` → 0, `Black` → 1, `Other` → 2).
  - **Label Encoding**: `LabelEncoder` assigns integers to categories in other columns:
    - `Marital Status`: e.g., `Married` → 1, `Divorced` → 0, etc.
    - `T Stage `: e.g., `T1` → 0, `T2` → 1, `T3` → 2, `T4` → 3.
    - `N Stage`: e.g., `N1` → 0, `N2` → 1, `N3` → 2.
    - `6th Stage`: e.g., `IIA` → 0, `IIB` → 1, etc.
    - `differentiate`: e.g., `Poorly differentiated` → 1, `Moderately differentiated` → 0.
    - `Grade`: Adjusted "anaplastic; Grade IV" to "4", then encoded (e.g., `3` → 3, `2` → 2, `1` → 1, `4` → 0).
    - `A Stage`: e.g., `Regional` → 1, `Distant` → 0.
    - `Estrogen Status` & `Progesterone Status`: e.g., `Positive` → 1, `Negative` → 0.
  - After encoding, `data.info()` shows all columns as integers (`int32` or `int64`), ready for modeling.

---

## **7. Defining Features and Target**
```python
X = data.drop('Status', axis=1)
Y = data['Status']
```
- **Purpose**: Separates the dataset into features (`X`) and target (`Y`).
- **What’s Happening**:
  - `X`: A DataFrame with 15 feature columns (e.g., `Age`, `Tumor Size`, `Survival Months`), excluding `Status`.
  - `Y`: A Series containing the `Status` column (1 for "Alive", 0 for "Dead").

---

## **8. Splitting the Data**
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```
- **Purpose**: Divides the data into training and testing sets.
- **What’s Happening**:
  - `test_size=0.2`: 20% (805 rows) for testing, 80% (3218 rows) for training.
  - `random_state=0`: Ensures reproducibility.
  - Outputs: `X_train`, `X_test` (features), `Y_train`, `Y_test` (targets).

---

## **9. Training the Logistic Regression Model**
```python
LGR = LogisticRegression()
LGR.fit(X_train, Y_train)
```
- **Purpose**: Initializes and trains a logistic regression model.
- **What’s Happening**:
  - The model learns from `X_train` and `Y_train` to predict `Status`.
  - A `ConvergenceWarning` appears, indicating the solver reached the default iteration limit (100). This suggests potential improvements like scaling features or increasing `max_iter`.

---

## **10. Evaluating on Training Data**
```python
Y_train_pred = LGR.predict(X_train)
pd.DataFrame({'Original_Y_test': Y_train, 'Predicted Y_test': Y_train_pred})
accuracy_score(Y_train, Y_train_pred)  # Output: 0.8965
confusion_matrix(Y_train, Y_train_pred)  # Output: [[218, 270], [63, 2667]]
print(classification_report(Y_train, Y_train_pred))
```
- **Purpose**: Assesses model performance on the training set.
- **What’s Happening**:
  - **Predictions**: `Y_train_pred` contains predicted values.
  - **Comparison**: A DataFrame shows actual vs. predicted values.
  - **Accuracy**: 0.8965 (89.65% correct predictions).
  - **Confusion Matrix**:
    - True Negatives (TN): 218 (correctly predicted "Dead").
    - False Positives (FP): 270 (predicted "Alive" but actually "Dead").
    - False Negatives (FN): 63 (predicted "Dead" but actually "Alive").
    - True Positives (TP): 2667 (correctly predicted "Alive").
  - **Classification Report**:
    ```
              precision    recall  f1-score   support
    0         0.78        0.45    0.57       488
    1         0.91        0.98    0.94      2730
    accuracy                      0.90      3218
    ```

---

## **11. Evaluating on Testing Data**
```python
Y_test_pred = LGR.predict(X_test)
pd.DataFrame({'Original_Y_test': Y_test, 'Predicted Y_test': Y_test_pred})
accuracy_score(Y_test, Y_test_pred)  # Output: 0.9006
confusion_matrix(Y_test, Y_test_pred)  # Output: [[63, 65], [15, 662]]
print(classification_report(Y_test, Y_test_pred))
```
- **Purpose**: Evaluates model performance on unseen test data.
- **What’s Happening**:
  - **Predictions**: `Y_test_pred` contains test set predictions.
  - **Accuracy**: 0.9006 (90.06% correct), slightly better than training, indicating good generalization.
  - **Confusion Matrix**:
    - TN: 63
    - FP: 65
    - FN: 15
    - TP: 662
  - **Classification Report**:
    ```
              precision    recall  f1-score   support
    0         0.81        0.49    0.61       128
    1         0.91        0.98    0.94       677
    accuracy                      0.90       805
    ```

---

## **12. Checking Class Distribution**
```python
data.Status.value_counts()  # Output: 1: 3407, 0: 616
```
- **Purpose**: Examines the balance of the target variable.
- **What’s Happening**: Reveals an imbalanced dataset:
  - "Alive" (1): 3407 instances (84.7%).
  - "Dead" (0): 616 instances (15.3%).
  - This imbalance explains high accuracy but lower recall for class 0 (0.49 on test data), as the model favors the majority class.

---

## **Summary**
This Jupyter Notebook demonstrates a complete pipeline for binary classification using logistic regression:
1. **Data Loading and Cleaning**: Loads "Breast_Cancer.csv", removes duplicates, and confirms no missing values.
2. **Preprocessing**: Encodes categorical variables (`Race`, `T Stage`, etc.) into numerical format.
3. **Modeling**: Trains a logistic regression model to predict `Status` (Alive/Dead).
4. **Evaluation**: Achieves ~90% accuracy on both training and test sets, though performance on "Dead" (0) is weaker due to class imbalance.

The model performs well overall but could be improved by addressing the convergence warning (e.g., scaling features with `StandardScaler`) and handling class imbalance (e.g., using `class_weight` or oversampling). This analysis provides a solid foundation for breast cancer survival prediction.
