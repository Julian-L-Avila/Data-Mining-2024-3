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
## Se entrena con valores pseudo aleatorios de $A, f, t$ y el correspondiente
## $ x = A\cos{2 \pi f t}$
#
## Input:
# - $A$
# - $f$
# - $t$

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)

def load_data(file_path):
  df = pd.read_csv(file_path, sep="\t").values
  return df[:, 1:4], df[:, 4]

def perceptron_train(X, y, activation, lr_values, epochs=100000):
  best_lr, best_w, best_b, lowest_mse = None, None, None, float('inf')
  cost_history = {}
  for lr in lr_values:
    w, b = np.random.randn(X.shape[1]), np.random.randn()
    costs = []
    for _ in range(epochs):
      y_pred = activation(X @ w + b)
      error = y - y_pred
      w += lr * (error @ X) / len(y)
      b += lr * error.mean()
      mse = np.mean(error ** 2)
      costs.append(mse)
      if mse < lowest_mse:
        best_lr, best_w, best_b, lowest_mse = lr, w, b, mse
        cost_history[lr] = costs
    return best_w, best_b, cost_history

def perceptron_predict(X, w, b, activation):
  return activation(X @ w + b)

X, y = load_data("./Data/train.tsv")
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

activations = {"Linear": lambda x: x, "ReLU": relu, "Tanh": np.tanh, "Sigmoid": sigmoid, "LeakyReLU": leaky_relu}
lr_values = [0.0001, 0.001, 0.01, 0.1]

models, cost_histories = {}, {}
for name, act in activations.items():
  w, b, cost_history = perceptron_train(X_train, y_train, act, lr_values)
  models[name] = (w, b)
  cost_histories[name] = cost_history

predictions = {name: perceptron_predict(X_test, w, b, act) for name, (w, b), act in zip(models, models.values(), activations.values())}
mse_values = {name: np.mean((y_test - y_pred) ** 2) for name, y_pred in predictions.items()}

for name, mse in mse_values.items():
  print(f"MSE ({name} Perceptron): {mse:.4f}")

for name, (w, b) in models.items():
  print(f"{name} Perceptron: Weights = {w}, Bias = {b:.4f}")

amplitude, frequency = 1, 1
t_values = np.linspace(0, 5, 1000)
th_position = amplitude * np.cos(2 * np.pi * frequency * t_values)

plt.figure(figsize=(10, 5))
for name, y_pred in predictions.items():
  plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', label="Ideal Fit")
plt.xlabel("Real Displacement")
plt.ylabel("Predicted Displacement")
plt.legend()
plt.title("Comparison of Perceptron Models (TSV Data)")
plt.show()

plt.figure(figsize=(10, 5))
for name, history in cost_histories.items():
  for lr, costs in history.items():
    plt.plot(costs, label=f"{name} LR={lr}")
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Cost Function over Epochs for Different Models and Learning Rates")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  y_sim = perceptron_predict(X_sim, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()

# %% [markdown]
## Se entrena con valores pseudo aleatoreos de $A, f, t, m, t$ y al correspondiente
## $x = A\cos{2\pi f t}$
#
## Input:
# - $A$
# - $f$
# - $t$
# - $m$

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)

def load_data(file_path):
  df = pd.read_csv(file_path, sep="\t").values
  return df[:, :4], df[:, 4]

def perceptron_train(X, y, activation, lr_values, epochs=100000):
  best_lr, best_w, best_b, lowest_mse = None, None, None, float('inf')
  cost_history = {}
  for lr in lr_values:
    w, b = np.random.randn(X.shape[1]), np.random.randn()
    costs = []
    for _ in range(epochs):
      y_pred = activation(X @ w + b)
      error = y - y_pred
      w += lr * (error @ X) / len(y)
      b += lr * error.mean()
      mse = np.mean(error ** 2)
      costs.append(mse)
      if mse < lowest_mse:
        best_lr, best_w, best_b, lowest_mse = lr, w, b, mse
        cost_history[lr] = costs
    return best_w, best_b, cost_history

def perceptron_predict(X, w, b, activation):
  return activation(X @ w + b)

X, y = load_data("./Data/train.tsv")
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

activations = {"Linear": lambda x: x, "ReLU": relu, "Tanh": np.tanh, "Sigmoid": sigmoid, "LeakyReLU": leaky_relu}
lr_values = [0.0001, 0.001, 0.01, 0.1]

models, cost_histories = {}, {}
for name, act in activations.items():
  w, b, cost_history = perceptron_train(X_train, y_train, act, lr_values)
  models[name] = (w, b)
  cost_histories[name] = cost_history

predictions = {name: perceptron_predict(X_test, w, b, act) for name, (w, b), act in zip(models, models.values(), activations.values())}
mse_values = {name: np.mean((y_test - y_pred) ** 2) for name, y_pred in predictions.items()}

for name, mse in mse_values.items():
  print(f"MSE ({name} Perceptron): {mse:.4f}")

for name, (w, b) in models.items():
  print(f"{name} Perceptron: Weights = {w}, Bias = {b:.4f}")

amplitude, frequency = 1, 1
mass = 10.0
t_values = np.linspace(0, 5, 1000)
th_position = amplitude * np.cos(2 * np.pi * frequency * t_values)

plt.figure(figsize=(10, 5))
for name, y_pred in predictions.items():
  plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', label="Ideal Fit")
plt.xlabel("Real Displacement")
plt.ylabel("Predicted Displacement")
plt.legend()
plt.title("Comparison of Perceptron Models (TSV Data)")
plt.show()

plt.figure(figsize=(10, 5))
for name, history in cost_histories.items():
  for lr, costs in history.items():
    plt.plot(costs, label=f"{name} LR={lr}")
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Cost Function over Epochs for Different Models and Learning Rates")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, mass), np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  y_sim = perceptron_predict(X_sim, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()

# %% [markdown]
## Se entrena con valores fijos $A, f$, con $t$ variable y al correspondiente
## $x = A\cos{2\pi f t}$
#
## Input:
# - $A$
# - $f$
# - $t$

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)

def load_data(file_path):
  df = pd.read_csv(file_path, sep="\t").values
  return df[:, :3], df[:, 3]

def perceptron_train(X, y, activation, lr_values, epochs=100000):
  best_lr, best_w, best_b, lowest_mse = None, None, None, float('inf')
  cost_history = {}
  for lr in lr_values:
    w, b = np.random.randn(X.shape[1]), np.random.randn()
    costs = []
    for _ in range(epochs):
      y_pred = activation(X @ w + b)
      error = y - y_pred
      w += lr * (error @ X) / len(y)
      b += lr * error.mean()
      mse = np.mean(error ** 2)
      costs.append(mse)
      if mse < lowest_mse:
        best_lr, best_w, best_b, lowest_mse = lr, w, b, mse
        cost_history[lr] = costs
    return best_w, best_b, cost_history

def perceptron_predict(X, w, b, activation):
  return activation(X @ w + b)

X, y = load_data("./Data/harmonic-oscillator.tsv")
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

activations = {"Linear": lambda x: x, "ReLU": relu, "Tanh": np.tanh, "Sigmoid": sigmoid, "LeakyReLU": leaky_relu}
lr_values = [0.0001, 0.001, 0.01, 0.1]

models, cost_histories = {}, {}
for name, act in activations.items():
  w, b, cost_history = perceptron_train(X_train, y_train, act, lr_values)
  models[name] = (w, b)
  cost_histories[name] = cost_history

predictions = {name: perceptron_predict(X_test, w, b, act) for name, (w, b), act in zip(models, models.values(), activations.values())}
mse_values = {name: np.mean((y_test - y_pred) ** 2) for name, y_pred in predictions.items()}

for name, mse in mse_values.items():
  print(f"MSE ({name} Perceptron): {mse:.4f}")

for name, (w, b) in models.items():
  print(f"{name} Perceptron: Weights = {w}, Bias = {b:.4f}")

amplitude, frequency = 1, 1
t_values = np.linspace(0, 5, 1000)
th_position = amplitude * np.cos(2 * np.pi * frequency * t_values)

plt.figure(figsize=(10, 5))
for name, y_pred in predictions.items():
  plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', label="Ideal Fit")
plt.xlabel("Real Displacement")
plt.ylabel("Predicted Displacement")
plt.legend()
plt.title("Comparison of Perceptron Models (TSV Data)")
plt.show()

plt.figure(figsize=(10, 5))
for name, history in cost_histories.items():
  for lr, costs in history.items():
    plt.plot(costs, label=f"{name} LR={lr}")
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Cost Function over Epochs for Different Models and Learning Rates")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  y_sim = perceptron_predict(X_sim, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()

# %% [markdown]
## Se entrena con valores pseudo aleatoreos $A, f, t$
## y el correspondiente $x = A \cos{2 \pi f t}$
#
## Input:
# - $A$
# - $f$
# - $\cos{2 \pi f t}$
# - $\sin{2 \pi f t}$

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation functions
relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)

def load_data(file_path):
  df = pd.read_csv(file_path, sep="\t").values
  return df[:, 1:4], df[:, 4]

def perceptron_train(X, y, activation, lr_values, epochs=500000):
  best_lr, best_w, best_b, lowest_mse = None, None, None, float('inf')
  cost_history = {}

  for lr in lr_values:
    w, b = np.random.randn(X.shape[1]), np.random.randn()
    costs = []

    for _ in range(epochs):
      y_pred = activation(X @ w + b)
      error = y - y_pred
      w += lr * (error @ X) / len(y)
      b += lr * error.mean()
      mse = np.mean(error ** 2)
      costs.append(mse)

      if mse < lowest_mse:
        best_lr, best_w, best_b, lowest_mse = lr, w, b, mse

        cost_history[lr] = costs

    return best_w, best_b, cost_history

def perceptron_predict(X, w, b, activation):
  return activation(X @ w + b)

# Load and split data
X, y = load_data("./Data/train.tsv")
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Transforming time into sin and cos components
def transform_features(X):
  return np.column_stack((
    X[:, 0],  # Amplitude
    X[:, 1],  # Frequency
    np.sin(2 * np.pi * X[:, 1] * X[:, 2]),  # sin(2πft)
    np.cos(2 * np.pi * X[:, 1] * X[:, 2])   # cos(2πft)
  ))

X_train_transformed = transform_features(X_train)
X_test_transformed = transform_features(X_test)

# Activation functions
activations = {
  "Linear": lambda x: x,
  "ReLU": relu,
  "Tanh": np.tanh,
  "Sigmoid": sigmoid,
  "LeakyReLU": leaky_relu
}
lr_values = [0.0001, 0.001, 0.01, 0.05, 0.1]

# Train models
models, cost_histories = {}, {}
for name, act in activations.items():
  w, b, cost_history = perceptron_train(X_train_transformed, y_train, act, lr_values)
  models[name] = (w, b)
  cost_histories[name] = cost_history

# Predictions
predictions = {
  name: perceptron_predict(X_test_transformed, *models[name], activations[name])
  for name in models
}

# Calculate MSE
mse_values = {
  name: np.mean((y_test - y_pred) ** 2)
  for name, y_pred in predictions.items()
}

# MSE Results
for name, mse in mse_values.items():
  print(f"MSE ({name} Perceptron): {mse:.4f}")

for name, (w, b) in models.items():
  print(f"{name} Perceptron: Weights = {w}, Bias = {b:.4f}")

# Theoretical harmonic position (for comparison)
amplitude, frequency = 1, 1
t_values = np.linspace(0, 5, 1000)
th_position = amplitude * np.cos(2 * np.pi * frequency * t_values)

# Scatter plot of predictions vs real values
plt.figure(figsize=(10, 5))
for name, y_pred in predictions.items():
  plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r', label="Ideal Fit")
plt.xlabel("Real Displacement")
plt.ylabel("Predicted Displacement")
plt.legend()
plt.title("Comparison of Perceptron Models (TSV Data)")
plt.show()

# Cost function over epochs (only best LR per model)
plt.figure(figsize=(10, 5))
for name, history in cost_histories.items():
  best_lr = min(history, key=lambda lr: history[lr][-1])  # Find LR with lowest final cost
  plt.plot(history[best_lr], label=f"{name} (LR={best_lr})")

plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Best Cost Function over Epochs for Each Model")
plt.show()

# Theoretical vs predicted position in harmonic motion
plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  X_sim_transformed = transform_features(X_sim)
  y_sim = perceptron_predict(X_sim_transformed, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")

plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  X_sim_transformed = transform_features(X_sim)
  y_sim = perceptron_predict(X_sim_transformed, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")

plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()

# %% [markdown]
## Se eliminan los metodos que se alejan por completo i.e $x \to \infty$

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation functions
relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)

def load_data(file_path):
  df = pd.read_csv(file_path, sep="\t").values
  return df[:, 1:4], df[:, 4]

def perceptron_train(X, y, activation, lr_values, epochs=500000):
  best_lr, best_w, best_b, lowest_mse = None, None, None, float('inf')
  cost_history = {}

  for lr in lr_values:
    w, b = np.random.randn(X.shape[1]), np.random.randn()
    costs = []

    for _ in range(epochs):
      y_pred = activation(X @ w + b)
      error = y - y_pred
      w += lr * (error @ X) / len(y)
      b += lr * error.mean()
      mse = np.mean(error ** 2)
      costs.append(mse)

      if mse < lowest_mse:
        best_lr, best_w, best_b, lowest_mse = lr, w, b, mse

    cost_history[lr] = costs

  return best_w, best_b, cost_history

def perceptron_predict(X, w, b, activation):
  return activation(X @ w + b)

# Load and split data
X, y = load_data("./Data/train.tsv")
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Transforming time into sin and cos components
def transform_features(X):
  return np.column_stack((
    X[:, 0],  # Amplitude
    X[:, 1],  # Frequency
    np.sin(2 * np.pi * X[:, 1] * X[:, 2]),  # sin(2πft)
    np.cos(2 * np.pi * X[:, 1] * X[:, 2])   # cos(2πft)
  ))

X_train_transformed = transform_features(X_train)
X_test_transformed = transform_features(X_test)

# Activation functions
activations = {
  "Linear": lambda x: x,
  "Tanh": np.tanh,
  "Sigmoid": sigmoid,
}
lr_values = [0.0001, 0.001, 0.01, 0.05, 0.1]

# Train models
models, cost_histories = {}, {}
for name, act in activations.items():
  w, b, cost_history = perceptron_train(X_train_transformed, y_train, act, lr_values)
  models[name] = (w, b)
  cost_histories[name] = cost_history

# Predictions
predictions = {
  name: perceptron_predict(X_test_transformed, *models[name], activations[name])
  for name in models
}

# Calculate MSE
mse_values = {
  name: np.mean((y_test - y_pred) ** 2)
  for name, y_pred in predictions.items()
}

# MSE Results
for name, mse in mse_values.items():
  print(f"MSE ({name} Perceptron): {mse:.4f}")

for name, (w, b) in models.items():
  print(f"{name} Perceptron: Weights = {w}, Bias = {b:.4f}")

# Theoretical harmonic position (for comparison)
amplitude, frequency = 2.0, 4.0
t_values = np.linspace(0, 5, 1000)
th_position = amplitude * np.cos(2 * np.pi * frequency * t_values)

# Scatter plot of predictions vs real values
plt.figure(figsize=(10, 5))
for name, y_pred in predictions.items():
  plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r', label="Ideal Fit")
plt.xlabel("Real Displacement")
plt.ylabel("Predicted Displacement")
plt.legend()
plt.title("Comparison of Perceptron Models (TSV Data)")
plt.show()

# Cost function over epochs (only best LR per model)
plt.figure(figsize=(10, 5))
for name, history in cost_histories.items():
  best_lr = min(history, key=lambda lr: history[lr][-1])  # Find LR with lowest final cost
  plt.plot(history[best_lr], label=f"{name} (LR={best_lr})")

plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Best Cost Function over Epochs for Each Model")
plt.show()

# Theoretical vs predicted position in harmonic motion
plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  X_sim_transformed = transform_features(X_sim)
  y_sim = perceptron_predict(X_sim_transformed, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")

plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()

# %% [markdown]
## Se entrena con valores fijos $A, f$, con $t$ variable y al correspondiente
## $x = A\cos{2\pi f t}$
#
## Input:
# - $A$
# - $f$
# - $\cos{2 \pi f t}$
# - $\sin{2 \pi f t}$

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation functions
relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)

def load_data(file_path):
  df = pd.read_csv(file_path, sep="\t").values
  return df[:, 1:4], df[:, 4]

def perceptron_train(X, y, activation, lr_values, epochs=500000):
  best_lr, best_w, best_b, lowest_mse = None, None, None, float('inf')
  cost_history = {}

  for lr in lr_values:
    w, b = np.random.randn(X.shape[1]), np.random.randn()
    costs = []

    for _ in range(epochs):
      y_pred = activation(X @ w + b)
      error = y - y_pred
      w += lr * (error @ X) / len(y)
      b += lr * error.mean()
      mse = np.mean(error ** 2)
      costs.append(mse)

      if mse < lowest_mse:
        best_lr, best_w, best_b, lowest_mse = lr, w, b, mse

    cost_history[lr] = costs

  return best_w, best_b, cost_history

def perceptron_predict(X, w, b, activation):
  return activation(X @ w + b)

# Load and split data
X, y = load_data("./Data/harmonic-oscillator.tsv")
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Transforming time into sin and cos components
def transform_features(X):
  return np.column_stack((
    X[:, 0],  # Amplitude
    X[:, 1],  # Frequency
    np.sin(2 * np.pi * X[:, 1] * X[:, 2]),  # sin(2πft)
    np.cos(2 * np.pi * X[:, 1] * X[:, 2])   # cos(2πft)
  ))

X_train_transformed = transform_features(X_train)
X_test_transformed = transform_features(X_test)

# Activation functions
activations = {
  "Linear": lambda x: x,
  "Tanh": np.tanh,
  "Sigmoid": sigmoid,
}
lr_values = [0.0001, 0.001, 0.01, 0.05, 0.1]

# Train models
models, cost_histories = {}, {}
for name, act in activations.items():
  w, b, cost_history = perceptron_train(X_train_transformed, y_train, act, lr_values)
  models[name] = (w, b)
  cost_histories[name] = cost_history

# Predictions
predictions = {
  name: perceptron_predict(X_test_transformed, *models[name], activations[name])
  for name in models
}

# Calculate MSE
mse_values = {
  name: np.mean((y_test - y_pred) ** 2)
  for name, y_pred in predictions.items()
}

# MSE Results
for name, mse in mse_values.items():
  print(f"MSE ({name} Perceptron): {mse:.4f}")

for name, (w, b) in models.items():
  print(f"{name} Perceptron: Weights = {w}, Bias = {b:.4f}")

# Theoretical harmonic position (for comparison)
amplitude, frequency = 2.0, 4.0
t_values = np.linspace(0, 5, 1000)
th_position = amplitude * np.cos(2 * np.pi * frequency * t_values)

# Scatter plot of predictions vs real values
plt.figure(figsize=(10, 5))
for name, y_pred in predictions.items():
  plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r', label="Ideal Fit")
plt.xlabel("Real Displacement")
plt.ylabel("Predicted Displacement")
plt.legend()
plt.title("Comparison of Perceptron Models (TSV Data)")
plt.show()

# Cost function over epochs (only best LR per model)
plt.figure(figsize=(10, 5))
for name, history in cost_histories.items():
  best_lr = min(history, key=lambda lr: history[lr][-1])  # Find LR with lowest final cost
  plt.plot(history[best_lr], label=f"{name} (LR={best_lr})")

plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Best Cost Function over Epochs for Each Model")
plt.show()

# Theoretical vs predicted position in harmonic motion
plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  X_sim_transformed = transform_features(X_sim)
  y_sim = perceptron_predict(X_sim_transformed, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")

plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()
# %% [markdown]
## Se entrena con valores pseudo aleatoreos $A, f, m, t$
## y el correspondiente $x = A \cos{2 \pi f t}$
#
## Input:
# - $A$
# - $f$
# - $\cos{2 \pi f t}$
# - $\sin{2 \pi f t}$
# - $m$

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation functions
relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)

def load_data(file_path):
  df = pd.read_csv(file_path, sep="\t").values
  return df[:, :4], df[:, 4]

def perceptron_train(X, y, activation, lr_values, epochs=500000):
  best_lr, best_w, best_b, lowest_mse = None, None, None, float('inf')
  cost_history = {}

  for lr in lr_values:
    w, b = np.random.randn(X.shape[1]), np.random.randn()
    costs = []

    for _ in range(epochs):
      y_pred = activation(X @ w + b)
      error = y - y_pred
      w += lr * (error @ X) / len(y)
      b += lr * error.mean()
      mse = np.mean(error ** 2)
      costs.append(mse)

      if mse < lowest_mse:
        best_lr, best_w, best_b, lowest_mse = lr, w, b, mse

    cost_history[lr] = costs

  return best_w, best_b, cost_history

def perceptron_predict(X, w, b, activation):
  return activation(X @ w + b)

# Load and split data
X, y = load_data("./Data/train.tsv")
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Transforming time into sin and cos components
def transform_features(X):
  return np.column_stack((
    X[:, 0],  # Amplitude
    X[:, 1],  # Amplitude
    X[:, 2],  # Frequency
    np.sin(2 * np.pi * X[:, 2] * X[:, 3]),  # sin(2πft)
    np.cos(2 * np.pi * X[:, 2] * X[:, 3])   # cos(2πft)
  ))

X_train_transformed = transform_features(X_train)
X_test_transformed = transform_features(X_test)

# Activation functions
activations = {
  "Linear": lambda x: x,
  "Tanh": np.tanh,
  "Sigmoid": sigmoid,
}
lr_values = [0.0001, 0.001, 0.01, 0.05, 0.1]

# Train models
models, cost_histories = {}, {}
for name, act in activations.items():
  w, b, cost_history = perceptron_train(X_train_transformed, y_train, act, lr_values)
  models[name] = (w, b)
  cost_histories[name] = cost_history

# Predictions
predictions = {
  name: perceptron_predict(X_test_transformed, *models[name], activations[name])
  for name in models
}

# Calculate MSE
mse_values = {
  name: np.mean((y_test - y_pred) ** 2)
  for name, y_pred in predictions.items()
}

# MSE Results
for name, mse in mse_values.items():
  print(f"MSE ({name} Perceptron): {mse:.4f}")

for name, (w, b) in models.items():
  print(f"{name} Perceptron: Weights = {w}, Bias = {b:.4f}")

# Theoretical harmonic position (for comparison)
amplitude, frequency, mass = 2.0, 2.0, 1
t_values = np.linspace(0, 5, 1000)
th_position = amplitude * np.cos(2 * np.pi * frequency * t_values)

# Scatter plot of predictions vs real values
plt.figure(figsize=(10, 5))
for name, y_pred in predictions.items():
  plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r', label="Ideal Fit")
plt.xlabel("Real Displacement")
plt.ylabel("Predicted Displacement")
plt.legend()
plt.title("Comparison of Perceptron Models (TSV Data)")
plt.show()

# Cost function over epochs (only best LR per model)
plt.figure(figsize=(10, 5))
for name, history in cost_histories.items():
  best_lr = min(history, key=lambda lr: history[lr][-1])  # Find LR with lowest final cost
  plt.plot(history[best_lr], label=f"{name} (LR={best_lr})")

plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Best Cost Function over Epochs for Each Model")
plt.show()

# Theoretical vs predicted position in harmonic motion
plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, mass), np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  X_sim_transformed = transform_features(X_sim)
  y_sim = perceptron_predict(X_sim_transformed, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")

plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()

# %% [markdown]
## Se entrena con valores fijos $A, f$, con $t$ variable y al correspondiente
## $x = A\cos{2\pi f t}$
#
## Input:
# - $A$
# - $f$
# - $\cos{2 \pi f t}$
# - $\sin{2 \pi f t}$

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation functions
relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)

def load_data(file_path):
  df = pd.read_csv(file_path, sep="\t").values
  return df[:, 1:4], df[:, 4]

def perceptron_train(X, y, activation, lr_values, epochs=500000):
  best_lr, best_w, best_b, lowest_mse = None, None, None, float('inf')
  cost_history = {}

  for lr in lr_values:
    w, b = np.random.randn(X.shape[1]), np.random.randn()
    costs = []

    for _ in range(epochs):
      y_pred = activation(X @ w + b)
      error = y - y_pred
      w += lr * (error @ X) / len(y)
      b += lr * error.mean()
      mse = np.mean(error ** 2)
      costs.append(mse)

      if mse < lowest_mse:
        best_lr, best_w, best_b, lowest_mse = lr, w, b, mse

    cost_history[lr] = costs

  return best_w, best_b, cost_history

def perceptron_predict(X, w, b, activation):
  return activation(X @ w + b)

# Load and split data
X, y = load_data("./Data/harmonic-oscillator.tsv")
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Transforming time into sin and cos components
def transform_features(X):
  return np.column_stack((
    X[:, 0],  # Amplitude
    X[:, 1],  # Frequency
    np.sin(2 * np.pi * X[:, 1] * X[:, 2]),  # sin(2πft)
    np.cos(2 * np.pi * X[:, 1] * X[:, 2])   # cos(2πft)
  ))

X_train_transformed = transform_features(X_train)
X_test_transformed = transform_features(X_test)

# Activation functions
activations = {
  "Linear": lambda x: x,
  "Tanh": np.tanh,
  "Sigmoid": sigmoid,
}
lr_values = [0.0001, 0.001, 0.01, 0.05, 0.1]

# Train models
models, cost_histories = {}, {}
for name, act in activations.items():
  w, b, cost_history = perceptron_train(X_train_transformed, y_train, act, lr_values)
  models[name] = (w, b)
  cost_histories[name] = cost_history

# Predictions
predictions = {
  name: perceptron_predict(X_test_transformed, *models[name], activations[name])
  for name in models
}

# Calculate MSE
mse_values = {
  name: np.mean((y_test - y_pred) ** 2)
  for name, y_pred in predictions.items()
}

# MSE Results
for name, mse in mse_values.items():
  print(f"MSE ({name} Perceptron): {mse:.4f}")

for name, (w, b) in models.items():
  print(f"{name} Perceptron: Weights = {w}, Bias = {b:.4f}")

# Theoretical harmonic position (for comparison)
amplitude, frequency = 2.0, 2.0
t_values = np.linspace(0, 5, 1000)
th_position = amplitude * np.cos(2 * np.pi * frequency * t_values)

# Scatter plot of predictions vs real values
plt.figure(figsize=(10, 5))
for name, y_pred in predictions.items():
  plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r', label="Ideal Fit")
plt.xlabel("Real Displacement")
plt.ylabel("Predicted Displacement")
plt.legend()
plt.title("Comparison of Perceptron Models (TSV Data)")
plt.show()

# Cost function over epochs (only best LR per model)
plt.figure(figsize=(10, 5))
for name, history in cost_histories.items():
  best_lr = min(history, key=lambda lr: history[lr][-1])  # Find LR with lowest final cost
  plt.plot(history[best_lr], label=f"{name} (LR={best_lr})")

plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Best Cost Function over Epochs for Each Model")
plt.show()

# Theoretical vs predicted position in harmonic motion
plt.figure(figsize=(10, 5))
plt.plot(t_values, th_position, label="Theoretical Position")
for name, (w, b) in models.items():
  X_sim = np.column_stack((np.full_like(t_values, amplitude), np.full_like(t_values, frequency), t_values))
  X_sim_transformed = transform_features(X_sim)
  y_sim = perceptron_predict(X_sim_transformed, w, b, activations[name])
  plt.plot(t_values, y_sim, label=f"{name} Prediction")

plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Theoretical vs Predicted Position in Harmonic Motion")
plt.show()
