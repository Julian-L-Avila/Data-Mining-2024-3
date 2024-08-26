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
## Filtering Data

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
df = pd.read_csv(url, sep=",")
df.head()

# %%
# Filter using conditions

df_filtered = df[(df["species"] == "virginica") | (df["sepal_length"] > 5.0)]
df_filtered.head()

# %%
df_filtered1 = df[df["species"].isin(["setosa", "versicolor"])]
df_filtered1

# %%
df_filtered2 = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
df_filtered2

# %%
corr_var = df_filtered2.corr()
print(corr_var)

# %%
plt.figure(figsize=(8,6))
plt.imshow(corr_var, cmap="coolwarm", interpolation="none")
plt.colorbar()

plt.xticks(np.arange(len(corr_var.columns)),corr_var.columns)
plt.yticks(np.arange(len(corr_var.columns)),corr_var.columns)

for i in range(len(corr_var.columns)):
    for j in range(len(corr_var.columns)):
        plt.text(j, i, f"{corr_var.iloc[i, j] : .2f}", ha="center", va="center", color="black")

plt.title("Correlation matrix of variables")
plt.tight_layout()
plt.show()

# %%
corr_var.dtypes

# %%
