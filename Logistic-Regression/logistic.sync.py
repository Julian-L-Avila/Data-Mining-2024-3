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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = "./diabetes.csv"

df = pd.read_csv(data)

df.head

# %%
x = 'Glucose'
y = 'DiabetesPedigreeFunction'
z = 'Outcome'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['green' if z == 0 else 'red' for z in df[z]]

ax.scatter(df[x], df[y], df[z], c=colors)

ax.set_xlabel('BP')
ax.set_ylabel('Age')
ax.set_zlabel('Outcome')
ax.set_title('3D Scatter Plot')

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Definir la función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función de entrenamiento de la regresión logística
def logistic_regression(X, y, lr, epochs):
    # Inicializar los parámetros del modelo
    m, n = X.shape
    theta = np.zeros(n)
    losses = []

    # Descenso de gradiente
    for epoch in range(epochs):
        # Calcular la predicción y la pérdida
        z = np.dot(X, theta)
        h = sigmoid(z)
        loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        losses.append(loss)

        # Calcular el gradiente
        gradient = np.dot(X.T, (h - y)) / m

        # Actualizar los parámetros
        theta -= lr * gradient

        # Imprimir la pérdida en cada época
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

    return theta, losses

# Datos de ejemplo
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([0, 0, 1, 1])

# Entrenar el modelo
learning_rate = 0.1
num_epochs = 1000
optimal_params, losses = logistic_regression(X, y, learning_rate, num_epochs)

# Plotear la curva de aprendizaje
plt.plot(range(1, num_epochs+1), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Curva de Aprendizaje')
plt.show()

print("Parámetros óptimos:", optimal_params)
