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

# %%
df_iris = pd.read_csv("./iris.csv")
df_iris.info()

# %%
df_iris.value_counts()

# %% [markdown]
# Realiza un conteo de los datos, mostrando cuantas veces se repiten.

# %%
len(df_iris)

# %% [markdown]
# Representa la cantidad de datos en el data frame como un int.

# %%
df_iris.nunique()

# %% [markdown]
# Cuenta el número de datos únicos en cada columna como un int.

# %%
df_iris.describe()

# %% [markdown]
# Realiza una descripción general de los datos, mostrando el conteo, el promedio,
# la desviación estándar, como también máximo y mínimo y los cuantil.

# %%
df_iris.sum()

# %% [markdown]
# Realiza la suma de todos los datos de cada columna.

# %%
df_iris.count()

# %% [markdown]
# Cuenta la cantidad de datos en cada serie.

# %%
df_iris.median(numeric_only=True)

# %% [markdown]
# Realiza el promedio aritmético de cada serie de datos.

# %%
df_iris.quantile([0.25, 0.75], numeric_only=True)

# %% [markdown]
# Regresa el valor medio de cada percentil.

# %%
def doble(x):
    return x + x

df_iris.apply(doble)

# %%
# Aplica la función a todos los datos.

# %%
df_iris.min()

# %% [markdown]
# Regresa el valor mínimo de cada serie.

# %%
df_iris.max()

# %% [markdown]
# Regresa el valor máximo de cada serie.

# %%
df_iris.mean(numeric_only=True)

# %% [markdown]
# Regresa el promedio de cada serie.

# %%
df_iris.var(numeric_only=True)

# %% [markdown]
# Regresa la varianza de cada serie.

# %%
df_iris.std(numeric_only=True)

# %% [markdown]
# Regresa la desviación estándar de cada serie.
