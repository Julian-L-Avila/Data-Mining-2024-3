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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = "./salaries.csv"

df = pd.read_csv('./salaries.csv')

df.info()

# %%
chunk_size = 200000

dataframes = []
nullcounts = 0.0

for chunk in pd.read_csv(data, chunksize=chunk_size):
    df = pd.DataFrame(chunk)

    print(df.info())
    dataframes.append(df)

    nullcounts += df.isnull().sum()

print(nullcounts)

# %%
print(len(dataframes))

dataframes[0].head(10)
