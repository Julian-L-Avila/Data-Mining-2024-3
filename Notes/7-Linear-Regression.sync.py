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
# Made by:
# - Julian Leonardo Avila Martinez - 20212107030
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = './Data-sets/Datos.xlsx'

# %%
dataframe =  pd.read_excel(data)
dataframe

# %%
df = dataframe.head(100)
df

# %%
x_column = 'ingresos'
y_column = 'consumo'

def hypothesis(w, x):
    return w * x

initial_w = 0.1
df['prediction'] = hypothesis(initial_w, df[x_column])
df

# %%
fid, ax = plt.subplots()

ax.scatter(df[x_column], df[y_column])
ax.plot(df[x_column], df['prediction'], color='red')

ax.set_xlabel(x_column)
ax.set_ylabel(y_column)

plt.grid(True)
plt.show()

# %%
def quadratic_cost(x, y, w):
    h = hypothesis(w, x)
    return np.mean((h - y) ** 2)

w_values = np.linspace(0.0, 0.3, 100)
error_values = np.array([quadratic_cost(df[x_column], df[y_column], w) for w in w_values])

w_min = w_values[error_values.argmin()]
error_min = error_values.min()

print(f"Optimal w: {w_min}, Minimum error: {error_min}")

plt.figure()
plt.plot(w_values, error_values, label='Quadratic Cost')
plt.scatter(w_min, error_min, color='red', label=f'Optimal w: {w_min}')
plt.xlabel('w')
plt.ylabel('Quadratic Cost')
plt.grid(True)
plt.legend()
plt.show()
plt.show()

# %%
def normal_equation(data, x, y):
    return np.sum(data[x] * data[y]) / np.sum(data[x]**2)

w_normal = normal_equation(df, x_column, y_column)

print(f"Using normal equation w =", w_normal)
print(f"Previous result w =", w_min)
print(f"Difference of:", w_normal - w_min)
print(f"Percentage relative error:", 100 * (w_normal - w_min) / w_normal)

plt.figure()
plt.scatter(df[x_column], df[y_column], label='Data')
plt.plot(df[x_column], df['prediction'], color='red', label=f'Initial Prediction (w={initial_w})')
plt.plot(df[x_column], hypothesis(w_normal, df[x_column]), color='purple', label=f'Final Prediction (w={w_normal})')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.grid(True)
plt.legend()
plt.show()

# %%
n_values = np.linspace(10, 200, 190, dtype=int)
test = []
train = []

for n in n_values:
    n_t = 1000 - n

    df_t = dataframe.head(n_t)
    df_n = df.head(n)

    w_normal = normal_equation(df_n, x_column, y_column)

    bias = quadratic_cost(df_n[x_column], df_n[y_column], w_normal)
    rp = quadratic_cost(df_t[x_column], df_t[y_column], w_normal)

    test.append(rp)
    train.append(bias)

plt.figure()
plt.plot(n_values, train, label="Train")
plt.plot(n_values, test, label="Test")

plt.xlabel("n")
plt.ylabel("Cost")
plt.title("Test and Train Cost vs. n")
plt.legend()
plt.grid(True)

plt.show()
