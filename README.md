# House Price Prediction – Regression Models (sklearn)

This repository demonstrates **end-to-end regression modeling** using `scikit-learn`, with a clean, reproducible workflow suitable for **GitHub review, interviews, and portfolio evaluation**.

The notebook compares:

* Linear Regression
* Support Vector Regression (SVR)
* Random Forest Regressor

with proper preprocessing, evaluation metrics, and best practices.

---

## 📁 Project Structure

```
├── Predicting_House_Prices.ipynb
├── README.md
└── requirements.txt
```

---

## 🧠 Key Concepts Covered

* Train / test split
* Feature scaling (when required)
* Residual analysis
* Model evaluation using multiple metrics
* Model comparison
* Best practices for serialization (pickle / pipeline)

---

## 📊 Evaluation Metrics Used

For **all regression models**, the following metrics are computed:

* **MSE** – Mean Squared Error (lower is better)
* **RMSE** – Root Mean Squared Error
* **MAE** – Mean Absolute Error
* **MAPE** – Mean Absolute Percentage Error *(only when target has no zeros)*
* **R² Score** – Variance explained (higher is better)
* **Adjusted R²** – Penalized R² for feature count

---

## 🔹 Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

regression = LinearRegression()
regression.fit(X_train_norm, y_train)

y_pred_lr = regression.predict(X_test_norm)

# Metrics
mse  = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred_lr)
r2   = r2_score(y_test, y_pred_lr)

n = X_test_norm.shape[0]
p = X_test_norm.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

✔ Interpretable coefficients
✔ Strong baseline
✖ Limited to linear relationships

---

## 🔹 Support Vector Regression (SVR)

> **Note:** SVR is highly scale-sensitive. Scaling is mandatory.

```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_norm, y_train)

y_pred_svr = svr.predict(X_test_norm)
```

✔ Captures non-linear patterns
✖ No interpretable coefficients
✖ Sensitive to hyperparameters

---

## 🔹 Random Forest Regressor

> **Note:** Tree-based models do **not** require feature scaling.

```python
from sklearn.ensemble import RandomForestRegressor

model_rfr = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_rfr.fit(X_train, y_train)

y_pred_rfr = model_rfr.predict(X_test)
```

✔ Handles non-linearity well
✔ Robust to outliers
✖ Less interpretable than linear models

---

## 📈 Unified Metrics Block (Reusable)

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
import numpy as np

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

---

## 💾 Model Serialization (Best Practice)

Always serialize the **entire pipeline**, not just the model.

```python
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

pipeline.fit(X_train, y_train)

with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
```

---

## ⚠️ Important Notes

* Do **not** scale data for Random Forest
* Do **not** use MAPE if `y` contains zeros
* Never `fit()` transformers on test data
* Always evaluate residuals, not just R²

---

## 🎯 Intended Audience

* Data science students
* ML interview preparation
* Portfolio / GitHub showcase
* Applied regression learning

---

## 📌 Verdict

This project demonstrates **correct ML hygiene**, not just working code.
It prioritizes:

* Reproducibility
* Metric integrity
* Model comparison
* Deployment awareness

If you can explain everything here clearly, you’re no longer a beginner.
