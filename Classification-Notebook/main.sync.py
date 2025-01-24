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

