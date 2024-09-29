# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# %%
iris = load_iris()
X = iris.data[:, :1]
y = iris.data[:, 1]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
model = LinearRegression()

model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

quadratic_error = (y_pred - y_test)**2

print(quadratic_error.mean())

# %%
plt.scatter(X_test, y_test, color='blue', label='Real Data')
plt.plot(X_test, y_pred, color='red', label='Prediction')
plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# %%
print(f'Coefficient: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')
