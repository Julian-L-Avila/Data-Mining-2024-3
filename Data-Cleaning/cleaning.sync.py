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

amount_data = 1978155

# %%
chunk_size = 200000

dataframes = []
nullcounts = 0.0

for chunk in pd.read_csv(data, chunksize=chunk_size):
    df_aux = pd.DataFrame(chunk)

    print(df_aux.info())
    dataframes.append(df_aux)

    nullcounts += df_aux.isnull().sum()

print(nullcounts)

# %%
print(len(dataframes))

dataframes[0].head(10)

# %%
missing_names = 31618
unique_names_set = set()
for df_aux in dataframes:
    unique_names_set.update(df_aux['nombre'].unique())

amount_data - (len(unique_names_set) + missing_names)

# %%
df_aux = df[df['nombre'].duplicated()]

df_aux = df_aux.dropna(subset=['nombre'])

df_aux

# %%
df_aux = df_aux[df_aux['nombre'] == "RAUL MARTIN ORTEGA PINEDO"]
df_aux

# %%
df_names = df

df_names['nombre'] = df_names['nombre'].fillna('RESEVADO')

print(df_names.isna().sum())

df_names

# %%
names_frequency = df_names['nombre'].value_counts()

#plt.figure(figsize=(10, 6))
#names_frequency.plot(kind='bar', color='skyblue')
#plt.title('Frequency of Names')
#plt.xlabel('Names')
#plt.ylabel('Frequency')
#plt.show()

# %%
df_names.head(20)

# %%
df_names[['denominacion', 'cargo']]

# %%
df_names[['denominacion', 'cargo']].isna().sum()

# %%
df_cargo = df_names

df_cargo['cargo'] = df_cargo['cargo'].fillna(df_cargo['denominacion'])

df_cargo[['denominacion', 'cargo']].isna().sum()

# %%
null_cargo = df_names[df_names['cargo'].isnull()]
null_cargo

# %%
null_cargo_monto = null_cargo[null_cargo['montoneto'].isnull()]
null_cargo_monto[['denominacion', 'montobruto']].isna().sum()
null_cargo_monto

# %%
null_cargo_monto = null_cargo_monto.dropna(subset=['montobruto'])
null_cargo_monto.isna().sum()

# %%
df_cargo = df_cargo.dropna(subset=['denominacion', 'cargo'], thresh=2)

df_cargo.isna().sum()
