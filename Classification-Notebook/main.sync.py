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
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# %%
# Load the data into a DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# %%
iris_df.head()

# %%
iris_df.describe()

# %%
# Pairplot to visualize the relationships between features
sns.pairplot(iris_df, hue='target', palette='Set1', markers=['o', 's', 'D'])
plt.suptitle("Pairplot of Iris Dataset Features", y=1.02)
plt.show()

# %%
# Visualizing 2D scatter plots for pairs of features
# Sepal Length vs Sepal Width
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x=iris.feature_names[0], y=iris.feature_names[1], hue='target', palette='Set1')
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Petal Length vs Petal Width
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x=iris.feature_names[2], y=iris.feature_names[3], hue='target', palette='Set1')
plt.title('Petal Length vs Petal Width')
plt.show()

# %%
# %%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# %%
# 3D plot for Sepal Length, Sepal Width, and Petal Length
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the points
scatter = ax.scatter(iris_df[iris.feature_names[0]], iris_df[iris.feature_names[1]], iris_df[iris.feature_names[2]],
                     c=iris_df['target'], cmap='Set1', marker='o')

# Adding labels
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])
plt.title('3D Plot: Sepal Length, Sepal Width, and Petal Length')

# Show the plot
plt.show()

# %%
# 3D plot for Sepal Length, Petal Length, and Petal Width
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the points
scatter = ax.scatter(iris_df[iris.feature_names[0]], iris_df[iris.feature_names[2]], iris_df[iris.feature_names[3]],
                     c=iris_df['target'], cmap='Set1', marker='o')

# Adding labels
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[2])
ax.set_zlabel(iris.feature_names[3])
plt.title('3D Plot: Sepal Length, Petal Length, and Petal Width')

# Show the plot
plt.show()


# %% [markdown]
## Logistic Regression
# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    RocCurveDisplay,
)

# %% Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# %% Function for binary classification with visualizations
def binary_classification(X, y, positive_class, class_name):
    y_binary = (y == positive_class).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, target_names=["Not " + class_name, class_name]),
    }

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not " + class_name, class_name], yticklabels=["Not " + class_name, class_name])
    plt.title(f"Confusion Matrix: {class_name} vs Not {class_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Plot ROC curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve: {class_name} vs Not {class_name}")
    plt.show()

    return metrics, model

# %% Function to plot decision boundaries (2D only)
def plot_decision_boundary(X, y, model, positive_class, class_name, feature_x, feature_y):
    # Reduce to 2D (two selected features)
    X_2d = X[:, [feature_x, feature_y]]
    y_binary = (y == positive_class).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X_2d, y_binary, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)

    # Generate grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_binary, edgecolors="k", cmap="coolwarm")
    plt.title(f"Decision Boundary: {class_name} vs Not {class_name}")
    plt.xlabel(iris.feature_names[feature_x])
    plt.ylabel(iris.feature_names[feature_y])
    plt.show()

# %% Binary classification for Setosa vs Not Setosa
print("Classification: Setosa vs Not Setosa")
setosa_metrics, setosa_model = binary_classification(X, y, positive_class=0, class_name="Setosa")
print(f"Accuracy: {setosa_metrics['accuracy']}")
print(f"F1 Score: {setosa_metrics['f1_score']}")
print("Confusion Matrix:")
print(setosa_metrics['confusion_matrix'])
print("Classification Report:")
print(setosa_metrics['classification_report'])

# Visualize decision boundary for Setosa vs Not Setosa (sepal length vs sepal width)
plot_decision_boundary(X, y, LogisticRegression(max_iter=1000), 0, "Setosa", 0, 1)

# %% Binary classification for Virginica vs Not Virginica
print("\nClassification: Virginica vs Not Virginica")
virginica_metrics, virginica_model = binary_classification(X, y, positive_class=2, class_name="Virginica")
print(f"Accuracy: {virginica_metrics['accuracy']}")
print(f"F1 Score: {virginica_metrics['f1_score']}")
print("Confusion Matrix:")
print(virginica_metrics['confusion_matrix'])
print("Classification Report:")
print(virginica_metrics['classification_report'])

# %% [markdown]
## Multiclass
# %% Imports and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
X, y = data.data, data.target

# %% Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=200).fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nMatriz de confusión:\n', conf_matrix)
print('\nReporte de clasificación:\n', class_report)

plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.7)
plt.title("Matriz de Confusión")
plt.colorbar()
plt.show()

# %% Decision Boundary Plotting
X = X.iloc[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor="k", cmap=plt.cm.Paired)
plt.title("Límite de decisión (Logistic Regression)")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.show()

# %% [markdown]
## SVM
# %% Imports and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
X, y = data.data, data.target

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
best_kernel = None
best_accuracy = 0

# %% Loop Over Kernels to Train and Evaluate SVM
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, kernel in enumerate(kernels):
    X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, :2], y, test_size=0.3, random_state=42)
    svm_model = SVC(kernel=kernel).fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_kernel = kernel

    # Decision Boundary Plot
    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[idx].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    scatter = axes[idx].scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, edgecolor="k", cmap=plt.cm.Paired, s=50)
    axes[idx].set_title(f"Decision Boundary ({kernel} kernel)")
    axes[idx].set_xlabel(data.feature_names[0])
    axes[idx].set_ylabel(data.feature_names[1])

    # Add legend to each plot
    handles, labels = scatter.legend_elements()
    axes[idx].legend(handles, labels, title="Classes")

plt.tight_layout()
plt.show()

# %% Model Evaluation for Best Kernel
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svm_model = SVC(kernel=best_kernel).fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Best Kernel: {best_kernel}')
print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nMatriz de confusión:\n', conf_matrix)
print('\nReporte de clasificación:\n', class_report)

fig, ax = plt.subplots()
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.7)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.title("Matriz de Confusión")
plt.colorbar(ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.7))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %% [markdown]
## Decision Tree
# %% Imports and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
data = load_iris(as_frame=True)
X, y = data.data, data.target

# %% Train and Evaluate Decision Tree Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=42)

tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=data.target_names)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nClassification Report:')
print(class_report)

# %% Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=data.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# %% Decision Tree Plot
plt.figure(figsize=(12, 8))
plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=data.target_names, rounded=True, proportion=False)
plt.title("Decision Tree (Depth=2)", fontsize=16)
plt.show()

# %% Train and Evaluate Decision Tree Model with Various Depths
depths = range(1, 21)  # Check tree depths from 1 to 20
train_accuracies = []
test_accuracies = []

for depth in depths:
    tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=42)
    tree_model.fit(X_train, y_train)

    # Train and test accuracy
    train_acc = accuracy_score(y_train, tree_model.predict(X_train))
    test_acc = accuracy_score(y_test, tree_model.predict(X_test))

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Find the best depth
best_depth = depths[np.argmin(abs(np.array(test_accuracies) - np.array(train_accuracies)))]

print(f'Best depth: {best_depth}')
print(f'Accuracy at best depth: {test_accuracies[best_depth - 1] * 100:.2f}%')

# %% Plot Accuracies as Function of Depth
plt.figure(figsize=(8, 6))
plt.plot(depths, train_accuracies, label="Train Accuracy", marker='o')
plt.plot(depths, test_accuracies, label="Test Accuracy", marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Tree Depth')
plt.legend()
plt.grid(True)
plt.show()

# %% Final Model with Best Depth
tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

# Accuracy and confusion matrix for the final model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=data.target_names)

print(f'Accuracy at best depth: {accuracy * 100:.2f}%')
print('\nClassification Report:')
print(class_report)

# %% Plot Confusion Matrix for Final Model
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=data.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Depth={best_depth})")
plt.show()

# %% Decision Tree Plot for Best Depth
plt.figure(figsize=(12, 8))
plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=data.target_names, rounded=True, proportion=False)
plt.title(f"Decision Tree (Depth={best_depth})", fontsize=16)
plt.show()

# %% [markdown]
## KNN
# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %%
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_knn(k_values):
    results = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_pred = knn.predict(X_train)
        test_pred = knn.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        diff = abs(train_acc - test_acc)

        results.append((k, train_acc, test_acc, diff))
    return results

# %%
results = evaluate_knn(range(1, 21))

best_train = max(results, key=lambda x: x[1])
best_test = max(results, key=lambda x: x[2])
lowest_diff = min(results, key=lambda x: x[3])

optimal_k = lowest_diff[0]
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Store classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

# %%
# Plot Accuracy vs. k-value for Train and Test data
k_values = range(1, 21)
train_accuracies = [train_acc for _, train_acc, _, _ in results]
test_accuracies = [test_acc for _, _, test_acc, _ in results]

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, label='Train Accuracy', marker='o', linestyle='-', color='b')
plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='o', linestyle='-', color='r')
plt.title('Train vs Test Accuracy for Different k-values')
plt.xlabel('k-value (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

print(optimal_k)
# %%
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %% [markdown]
## Naïve Bayes
# %% Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %% Load Iris dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# %% Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% Grid search to find the best hyperparameters (in this case, smoothing parameter for GaussianNB)
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# %% Best hyperparameters
print(f"Best hyperparameter: {grid_search.best_params_}")

# %% Train the model with the best parameters
best_nb = grid_search.best_estimator_
y_pred = best_nb.predict(X_test)

# %% Classification report and metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# %% Plot accuracy vs. var_smoothing parameter
results = grid_search.cv_results_
plt.figure(figsize=(8, 6))
plt.semilogx(results['param_var_smoothing'], results['mean_test_score'])
plt.xlabel('var_smoothing')
plt.ylabel('Mean Accuracy')
plt.title('Accuracy vs. var_smoothing')
plt.grid(True)
plt.show()

# %% Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# ## Hyperparámetros:

# %% [markdown]
# ### 1. Logistic Regresion:

# %% [markdown]
# #### 1.1. C: (inversa de la regularización). (Por defecto: '1.0')
#
# #### Descripción:
# Controla la regularización L2 en el modelo. Un valor alto de C significa menos regularización (el modelo se ajustará más a los datos), mientras que un valor bajo de C significa más regularización (el modelo será más simple y generalizará mejor).

# %%
model = LogisticRegression(C=1.0)

# %% [markdown]
# #### 1.2. solver: Método de optimización. (Por defecto: 'lbfgs')
#
# #### Descripción: 
# Especifica el algoritmo utilizado para optimizar la función de coste de la regresión logística.
#
# #### Opciones comunes:
# - 'liblinear': Un algoritmo eficiente para pequeños conjuntos de datos y modelos binarios.
# - 'newton-cg': Utiliza el método de Newton para optimizar, bueno para problemas más grandes y multiclasificación.
# - 'lbfgs': Un optimizador quasi-Newton que funciona bien para problemas más grandes.
# - 'saga': Algoritmo de optimización que también es eficiente con grandes datasets y permite la regularización elastic net.

# %%
model = LogisticRegression(solver='lbfgs')

# %% [markdown]
# #### 1.3. penalty: Tipo de regularización. (Por defecto: 'l2')
#
# #### Descripción: 
# Tipo de regularización utilizada en la regresión logística. En general, los dos tipos más comunes son:
#
# - L2: Regularización de Ridge, que es la más común y la que se usa por defecto.
# - L1: Regularización de Lasso, que puede inducir una sparsidad (es decir, hacer que algunos coeficientes del modelo sean cero).
# - elasticnet: Combinación de L1 y L2, útil cuando se desea un equilibrio entre ambos.

# %%
model = LogisticRegression(penalty='l2')

# %% [markdown]
# #### 1.4. max_iter: Número máximo de iteraciones para la optimización. (Por defecto: 100)
#
# #### Descripción: 
# Número máximo de iteraciones que el algoritmo puede realizar durante la optimización.

# %%
model = LogisticRegression(max_iter=100)

# %% [markdown]
# #### 1.5. tol: determina la precisión de la convergencia en el proceso de optimización. (Por defecto: 1e-4)
#
# #### Descripción: 
# Determina la tolerancia para la convergencia del optimizador. Si la mejora entre iteraciones sucesivas es menor que esta tolerancia, el algoritmo se detiene.

# %%
model = LogisticRegression(tol=1e-4)

# %% [markdown]
# ### 2. SVM:

# %% [markdown]
# #### 2.1. C: (parametro de penalización) Controla el margen y los errores. (Por defecto: C=1.0)
#
# #### Descripción: 
# Similar al de la regresión logística, C es un parámetro que controla el equilibrio entre el margen máximo (generalización) y los errores de clasificación en el conjunto de entrenamiento.

# %%
model = SVC(C=1.0)

# %% [markdown]
# #### 2.2. kernel: Funcion nucleo (Por defecto: 'rbf')
#
# #### Descripción: 
# Define el tipo de función de núcleo que se usa para transformar los datos en un espacio de características de mayor dimensión, donde se pueden encontrar los márgenes óptimos.
#
# ##### tipos de kernel:
# - 'linear': El núcleo lineal es adecuado para problemas donde las clases son separables de manera lineal.
# - 'poly': El núcleo polinómico se usa cuando las clases no son lineales pero pueden separarse mediante un polinomio.
# - 'rbf' (Radial Basis Function): Es el núcleo más común, utilizado cuando las clases son no lineales y separables mediante una función gaussiana.
# - 'sigmoid': Menos común, utilizado para problemas no lineales con una frontera de decisión sigmoidea.

# %%
model = SVC(kernel='rbf')

# %% [markdown]
# #### 2.3. gamma: Parámetro del núcleo. (Por defecto: 'scale')
#
# #### Descripción: 
# El parámetro gamma controla el alcance de la influencia de un solo punto de datos sobre la clasificación.

# %%
model = SVC(gamma='scale')

# %% [markdown]
# #### 2.4. degree: Grado del polinomio para el kernel 'poly'. (Por defecto: degree=3)
#
# #### Descripción: 
# Este parámetro es relevante solo cuando se usa el núcleo polinómico ('poly'). Especifica el grado del polinomio utilizado para mapear los datos al espacio de características.

# %%
model = SVC(kernel='poly', degree=3)

# %% [markdown]
# #### 2.5. shrinking: Usar heurística de "shrinking". (Por defecto: shrinking=True)
#
# #### Descripción: 
# Este parámetro indica si se debe utilizar la heurística de "shrinking" durante la optimización del SVM.
#
# - 'True': Aplica la heurística de "shrinking", lo que puede hacer que el algoritmo converja más rápido al eliminar ciertos puntos durante el entrenamiento.
# - 'False': No se aplica la heurística de "shrinking", lo que puede hacer el proceso más lento pero más exhaustivo.

# %%
model = SVC(shrinking=True)

# %% [markdown]
# ### 3. Decision Tree:

# %% [markdown]
# #### 3.1. max_depth: Profundidad máxima del árbol. (Por defecto: max_depth=None)
#
# #### Descripción: 
# Este parámetro controla la profundidad máxima del árbol de decisión, es decir, cuántos niveles de nodos puede tener el árbol.
#
# - Un valor alto de max_depth permite que el árbol crezca más y capture patrones más complejos, lo que puede llevar a sobreajuste si es demasiado alto.
# - Un valor bajo de max_depth limita el crecimiento del árbol y hace que el modelo sea más general y menos propenso a sobreajustar.
#
# #### Valor por defecto: 
# max_depth=None (lo que significa que el árbol se expandirá hasta que cada hoja contenga menos de min_samples_split muestras).

# %%
model = DecisionTreeClassifier(max_depth=5)

# %% [markdown]
# #### 3.2. min_samples_split: Número mínimo de muestras requeridas para dividir un nodo. (Por defecto: min_samples_split=2)
#
# #### Descripción: 
# Especifica el número mínimo de muestras requeridas para dividir un nodo interno.
#
# - Un valor bajo de min_samples_split permitirá que el árbol se divida en nodos más pequeños, lo que puede llevar a sobreajuste.
# - Un valor alto de min_samples_split hará que el árbol no divida los nodos a menos que haya un número suficiente de muestras, lo que hace que el modelo sea más generalizable.

# %%
model = DecisionTreeClassifier(min_samples_split=10)

# %% [markdown]
# #### 3.3. min_samples_leaf: Número mínimo de muestras en una hoja. (Por defecto: min_samples_leaf=1)
#
# #### Descripción: 
# Controla el número mínimo de muestras que debe tener una hoja para considerarse válida.
# - Un valor bajo de min_samples_leaf puede generar muchas hojas pequeñas, lo que puede llevar a sobreajuste.
# - Un valor alto hace que las hojas contengan más muestras y, por lo tanto, puede reducir el sobreajuste.

# %%
model = DecisionTreeClassifier(min_samples_leaf=1)

# %% [markdown]
# #### 3.4. max_features: Número máximo de características a considerar en cada división. (Por defecto: max_features=None)
#
#
# #### Descripción: 
# Especifica el número máximo de características que se deben considerar al buscar la mejor división para un nodo.
#
# - Si se ajusta a un valor pequeño, reduces la complejidad del modelo y puedes ayudar a evitar el sobreajuste, pero también puede reducir la precisión.
# - Si se establece como 'auto' o 'sqrt', utiliza la raíz cuadrada del número total de características. 'log2' usa el logaritmo en base 2 del número de características.

# %%
model = DecisionTreeClassifier(max_features='None')

# %% [markdown]
# #### 3.5. criterion: Función de calidad de la división. (Por defecto: criterion='gini')
#
# #### Descripción: 
# Controla la función de evaluación utilizada para medir la calidad de las divisiones en el árbol.
#
# - 'gini': Usa el índice de Gini, que mide la pureza de las divisiones. Es el valor por defecto.
# - 'entropy': Usa la entropía, que mide la incertidumbre o la impureza de las divisiones.

# %%
model = DecisionTreeClassifier(criterion='gini')

# %% [markdown]
# ### KNN
# - K-Nearest Neighbors (KNN): n_neighbors: The number of neighbors to use for knearest neighbors classification.
# - weights: Strategy for weighting the neighbors, options include 'uniform' (all points are weighted equally) or 'distance' (points closer to the query point are weighted more heavily).
# - algorithm: Algorithm used to compute the nearest neighbors, options include 'auto', 'ball_tree', 'kd_tree', or 'brute'.
# - leaf_size: The leaf size passed to the BallTree or KDTree algorithm.
# - p: Power parameter for the Minkowski distance metric. When p = 1, it’s equivalent to Manhattan distance, and when p = 2, it’s Euclidean distance.
# - metric: The distance metric to use, options include 'minkowski', 'euclidean', 'manhattan', etc.

# %% [markdown]
# ### Naive Bayes:
# - alpha (for MultinomialNB): Smoothing parameter to prevent zero probabilities for features not seen in the training data.
# - fit_prior (for MultinomialNB, GaussianNB): Whether to learn class prior probabilities or assume uniform class priors.
# - class_prior (for MultinomialNB, GaussianNB): Prior probabilities of the classes. If set to None, it is automatically computed from the training data.
