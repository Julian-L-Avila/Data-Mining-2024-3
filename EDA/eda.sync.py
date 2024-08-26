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
# # EDA
# Julian Leonardo Avila Martinez
# 20212107030

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv"
df = pd.read_csv(url, sep=";")

df.head()

# %%
df.shape

# %%
df.columns

# %%
df.info()

# %% [markdown]
# The data shows a list of 252 countries with information about certain aspects
# such as Name, Capital, Continent, Area, Codes for telephone communication,
# shipping, internet extension; Languages spoken on the territory and more.
#
# One can see that the data types seem to be consistent, being quantities the
# expected features, one may expect the phone to be a number type, however due
# to "+" extension it is treated as an object or string.
#
# There is a big amount of missing data, specially for the postal services.
# Some capitals are missing, some continents and neighbours are missing as well
# this could be due to being island nation without a clear continent of pertinence
# and the lack of land borders with other nations.
#
# Most notably there is one one country that has an equivalent flip code.

# %%
df.describe()

# %% [markdown]
# The numeric info one has is:
# - Area - Land are by its territory
# - GeoName ID - Number that identifies the location
# - Numeric - Numeric code that identifies country for communications
# - Population - Amount of inhabitants
#
# Due to the nature of the data, the only ones that could be related are the
# area and the population, as countries with larger area are more likely to have
# a bigger population.

# %%
df_numeric = df[["area", "geoname_id", "numeric", "population"]]
df_numeric

# %%
corr_var = df_numeric.corr()
print(corr_var)
