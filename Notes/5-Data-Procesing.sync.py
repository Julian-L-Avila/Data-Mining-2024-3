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

# %% [markdown]
# Names:
# Julian Avila - 20212107030
# Juan Acuña - 20212107034

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# %%
iris = datasets.load_iris()
type(iris)

# %%
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df.dtypes

# %%
corr_var = df.corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr_var, cmap="coolwarm")
fig.colorbar(cax)

# %%
plt.scatter(df["petal length (cm)"], df["petal width (cm)"])

m = np.linspace(0.2, 0.4, 100)
b = np.linspace(-1.0, 1.0, 100)

for i in m:
    for j in b:
        plt.plot(df["petal length (cm)"], i*df["petal length (cm)"]+j, color='r')

plt.title("Correlation between sepal width and length")
plt.xlabel("Petal length [cm]")
plt.ylabel("Petal width [cm]")
plt.show()

# %%

def RMSE(x, y, m, b):
    rmse = []
    n = len(x)
    for i in range(len(m)):
        for k in range(len(b)):
            err = 0.0
            for j in range(len(x)):
                y_pre = m[i] * x[j] + b[k]
                err += (y[j] - y_pre) * (y[j] - y_pre)
            rmse.append([np.sqrt(err / n), m[i], i, b[k], k])
    return rmse

rmse = RMSE(df["petal length (cm)"], df["petal width (cm)"], m, b)
print(min(rmse))

# %% [markdown]
# The algorithm used gives:
# - m = 0.4
# - b = -0.3131..
#
# m is the biggest possible value, so the first range for m must be higher.
# Then we are going to run the algorithm again but restricting the range for m
# and b to
# - m in (0.30, 0.50)
# - b in (-0.50, -0.20)

# %%
m = np.linspace(0.30, 0.50, 100)
b = np.linspace(-0.50, -0.20, 100)

rmse = RMSE(df["petal length (cm)"], df["petal width (cm)"], m, b)
print(min(rmse))

# %% [markdown]
# The algorithm used gives:
# - m = 0.4151...
# - b = -0.360606...

# %%
m = np.linspace(0.40, 0.43, 100)
b = np.linspace(-0.40, -0.31, 100)

rmse = RMSE(df["petal length (cm)"], df["petal width (cm)"], m, b)
print(min(rmse))

# %% [markdown]
# The new values given are:
# - m = 0.415757..
# - b = -0.362727...
#
# As the previous value is not that different from the new, we will consider
# these as the closest slope and intercept for the linear regression.

# %%
plt.scatter(df["petal length (cm)"], df["petal width (cm)"])

m = min(rmse)[1]
b = min(rmse)[3]

plt.plot(df["petal length (cm)"], m*df["petal length (cm)"] + b, linewidth=2.5,
        color='green')


plt.title("Correlation between sepal width and length")
plt.xlabel("Petal length [cm]")
plt.ylabel("Petal width [cm]")
plt.show()
