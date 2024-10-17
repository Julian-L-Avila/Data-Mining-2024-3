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

# %% [Markdown]
# Julian L. Avila - 20212107030
# Juan S. Acuna - 20212107034

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

data_1 = './Data/weatherHistory.csv'
data_2 = './Data/Solar_Prediction.csv'

df_1 = pd.read_csv(data_1)
df_2 = pd.read_csv(data_2)

# %%
print(df_1)
df_1.describe()

# %%
print(df_2)
df_2.describe()

# %%
def load_and_format_data(file_path, columns, date_column=None, unix_column=None):
    df = pd.read_csv(file_path)[columns]
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], utc=True)
    if unix_column:
        df[unix_column] = pd.to_datetime(df[unix_column], unit='s', utc=True)
    return df

def add_date_parts(df, date_column):
    date_parts = ['year', 'month', 'day', 'hour', 'minute']
    date_data = df[date_column].apply(lambda x: pd.Series([x.year, x.month, x.day, x.hour, x.minute]))
    return df.join(date_data.set_axis(date_parts, axis=1))

def rename_and_reorder(df, rename_map, new_order):
    return df.rename(columns=rename_map).reindex(columns=new_order)

data_1 = './Data/weatherHistory.csv'
data_2 = './Data/Solar_Prediction.csv'

columns_1 = ['Formatted Date', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)']
columns_2 = ['UNIXTime', 'Temperature', 'Humidity', 'WindDirection(Degrees)', 'Speed']

df_1 = load_and_format_data(data_1, columns_1, date_column='Formatted Date')
df_2 = load_and_format_data(data_2, columns_2, unix_column='UNIXTime')

df_1 = add_date_parts(df_1, 'Formatted Date')
df_2 = add_date_parts(df_2, 'UNIXTime')

rename_map_1 = {
    'Temperature (C)': 'temperature',
    'Humidity': 'humidity',
    'Wind Speed (km/h)': 'speed',
    'Wind Bearing (degrees)': 'direction'
}
rename_map_2 = {
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'WindDirection(Degrees)': 'direction',
    'Speed': 'speed'
}
new_columns = ['year', 'month', 'day', 'hour', 'minute', 'temperature', 'humidity', 'speed', 'direction']

df_1 = rename_and_reorder(df_1, rename_map_1, new_columns)
df_2 = rename_and_reorder(df_2, rename_map_2, new_columns)

df_1, df_2

# %%
def calculate_monthly_averages(df):
    return (
        df[['temperature', 'humidity', 'speed', 'direction']]
        .groupby(pd.Grouper(freq='ME'))
        .mean()
        .assign(
            year=lambda x: x.index.year,
            month=lambda x: x.index.month
        )
    )

def plot_temperature_collage(df, title, num_cols=3):
    unique_years = df['year'].unique()
    num_rows = (len(unique_years) + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(15, 5 * num_rows))
    grid = GridSpec(num_rows, num_cols, figure=fig)

    for idx, year in enumerate(unique_years):
        ax = fig.add_subplot(grid[idx // num_cols, idx % num_cols])
        yearly_data = df[df['year'] == year]
        scatter = ax.scatter(yearly_data['month'], yearly_data['temperature'],
                             c=yearly_data['humidity'], cmap='viridis')
        ax.set_title(f'{title} - {year}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Temperature')

    fig.colorbar(scatter, ax=ax, label='Humidity', orientation='horizontal')
    plt.tight_layout()
    plt.show()

def plot_temperature_collage_humidity(df, title, num_cols=3):
    unique_years = df['year'].unique()
    num_rows = (len(unique_years) + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(15, 5 * num_rows))
    grid = GridSpec(num_rows, num_cols, figure=fig)

    for idx, year in enumerate(unique_years):
        ax = fig.add_subplot(grid[idx // num_cols, idx % num_cols])
        yearly_data = df[df['year'] == year]
        scatter = ax.scatter(yearly_data['humidity'], yearly_data['temperature'],
                             c=yearly_data['month'], cmap='viridis')
        ax.set_title(f'{title} - {year}')
        ax.set_xlabel('Humidity')
        ax.set_ylabel('Temperature')

    fig.colorbar(scatter, ax=ax, label='Month', orientation='horizontal')
    plt.tight_layout()

def prepare_data(df, date_cols):
    df['datetime'] = pd.to_datetime(df[date_cols])
    df.set_index('datetime', inplace=True)
    return calculate_monthly_averages(df)

df_1 = prepare_data(df_1, ['year', 'month', 'day', 'hour', 'minute'])
df_2 = prepare_data(df_2, ['year', 'month', 'day', 'hour', 'minute'])

plot_temperature_collage(df_1, 'Weather History')
plot_temperature_collage(df_2, 'Solar Prediction')
plot_temperature_collage_humidity(df_1, 'Weather History')
plot_temperature_collage_humidity(df_2, 'Solar Prediction')

# %%
def run_linear_regression(df, target, feature):
    X = df[[feature]].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f'{target} vs {feature}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'R² Score: {r2_score(y_test, y_pred)}')

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, alpha=0.6, color='b')
    plt.plot(X_test, y_pred, color='r', lw=2)
    plt.xlabel(f'{feature}')
    plt.ylabel('Temperature')
    plt.title(f'Temperature vs {feature}')
    plt.show()

    return model

def run_multilinear_regression(df, target, features):
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f'{target} vs {features}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'R² Score: {r2_score(y_test, y_pred)}')

    return model

features = ['humidity', 'speed', 'direction', 'month']

for i in features:
    linear_model = run_linear_regression(df_1, target='temperature', feature=i)

multilinear_model = run_multilinear_regression(df_1, target='temperature', features=features)

# %%
df = df_1

df['month'] = df.index.month
features = ['month']
X = df[features]
y = df['temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(X_test['month'], y_test, alpha=0.6, color='b', label='Actual')
plt.scatter(X_test['month'], y_pred, color='r', alpha=0.6, label='Predicted')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Temperature vs Month (Actual vs Predicted)')
plt.legend()
plt.show()

features = ['month', 'humidity']
X = df[features]
y = df['temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(X_test['month'], y_test, alpha=0.6, color='b', label='Actual')
plt.scatter(X_test['month'], y_pred, color='r', alpha=0.6, label='Predicted')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Temperature vs Month (Actual vs Predicted)')
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['month'], X_test['humidity'], y_test, color='b', label='Actual')

month_range = np.linspace(X_test['month'].min(), X_test['month'].max(), 100)
humidity_range = np.linspace(X_test['humidity'].min(), X_test['humidity'].max(), 100)
month_grid, humidity_grid = np.meshgrid(month_range, humidity_range)

X_grid = pd.DataFrame({'month': month_grid.ravel(), 'humidity': humidity_grid.ravel()})
X_grid_poly = poly.transform(X_grid)

y_grid_pred = model.predict(X_grid_poly).reshape(month_grid.shape)

ax.plot_surface(month_grid, humidity_grid, y_grid_pred, color='r', alpha=0.5)

ax.set_xlabel('Month')
ax.set_ylabel('Humidity')
ax.set_zlabel('Temperature')
ax.set_title('3D plot of Temperature vs Month and Humidity')

scatter_proxy = plt.Line2D([0], [0], linestyle="none", marker='o', color='b')
surface_proxy = plt.Line2D([0], [0], linestyle="none", marker='s', color='r', alpha=0.5)
ax.legend([scatter_proxy, surface_proxy], ['Actual', 'Predicted Surface'])

plt.show()

model_1 = model

features = ['month', 'year', 'humidity']
X = df[features]
y = df['temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(X_test['month'], y_test, alpha=0.6, color='b', label='Actual')
plt.scatter(X_test['month'], y_pred, color='r', alpha=0.6, label='Predicted')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Temperature vs Month (Actual vs Predicted)')
plt.legend()
plt.show()

model_2 = model

features = ['month', 'humidity', 'speed', 'direction']
X = df[features]
y = df['temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(X_test['month'], y_test, alpha=0.6, color='b', label='Actual')
plt.scatter(X_test['month'], y_pred, color='r', alpha=0.6, label='Predicted')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Temperature vs Month (Actual vs Predicted)')
plt.legend()
plt.show()

model_3 = model

features = ['year', 'month', 'humidity', 'speed', 'direction']
X = df[features]
y = df['temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(X_test['month'], y_test, alpha=0.6, color='b', label='Actual')
plt.scatter(X_test['month'], y_pred, color='r', alpha=0.6, label='Predicted')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Temperature vs Month (Actual vs Predicted)')
plt.legend()
plt.show()

model_4 = model

# %%
df = df_1

df['month'] = df.index.month
features = ['humidity', 'speed', 'direction', 'month']
X = df[features]
y = df['temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', lw=2)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['humidity'], df['speed'], y, c=y, cmap='coolwarm', marker='o')

ax.set_xlabel('Humidity')
ax.set_ylabel('Speed')
ax.set_zlabel('Temperature')

plt.title('Humidity, Speed vs Temperature (3D Visualization)')
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='g')
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='r')
plt.xlabel('Predicted Temperature')
plt.ylabel('Residuals')
plt.title('Residuals of Polynomial Regression')
plt.show()
