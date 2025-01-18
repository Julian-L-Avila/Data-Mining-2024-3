# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="JtRLcznM2QAh"
# # Árboles de Decisión
#
# Veremos árboles de decisión y los conceptos subyacentes asociados.
#
# Haremos ejemplos de juguete y con datos generados artificialmente.

# %% id="3VdSI6EH2QAj"
import numpy as np
import matplotlib.pyplot as plt


# %% [markdown] id="G9w9iZBr2QAk"
# ## Entropía y Ganancia de Información

# %% [markdown] id="mBIPPzSU2QAk"
# ### Entropía
#
# Definamos entropía para una distribución probabilista:
#
# $$H(Y) = - \sum_{i=1}^k P(Y = y_i) log_2 P(Y = y_i)$$

# %% id="9BBkjfWb2QAk"
def entropy(probs):
    return - np.sum(probs * np.log2(probs))


# %% [markdown] id="en4mdoSa2QAl"
# Veamos posibles entropías para el problema de tirar una moneda adulterada:

# %% id="JW_RSUam2QAl" outputId="9119b2fc-abc8-46ef-f348-87dfdaaabf3a"
entropy(np.array([0.5, 0.5]))

# %% id="Wy5eUnnC2QAm" outputId="ce3ec021-e827-49f4-fc0e-3d301ad2667c"
entropy(np.array([0.01, 0.99]))

# %% id="pyTqDdlZ2QAm" outputId="014605ba-ea4c-4e8e-e37c-490be4ec6a60"
X = np.linspace(0, 1)[1:-1]
plt.plot(X, [entropy([x, 1-x]) for x in X])
plt.xlabel('P(Y=y_1)')
plt.ylabel('entropy')
plt.show()

# %% [markdown] id="8BZDB1gy2QAn"
# En el caso de dos monedas, tenemos cuatro combinaciones posibles, siendo $0.25$ la probabilidad de cada evento. Ejemplos:

# %% id="O6ScSc-i2QAn" outputId="2707c979-72e3-4a41-ce3c-962e391f8f75"
entropy(np.array([0.25, 0.25, 0.25, 0.25]))

# %% [markdown] id="FsdpZJLZ2QAo"
# Pero si las monedas estan sesgadas tendríamos:

# %% id="HvxCuAqL2QAo" outputId="f7541cc0-8a8f-4f02-8639-941fd57beea7"
entropy(np.array([0.49, 0.49, 0.01, 0.01]))


# %% [markdown] id="1296dW7_2QAp"
# ### Entropía de un Dataset
#
# Un dataset define una distribución empírica. La entropía del dataset es entones la entropía de la distribución asociada. Definamos el cálculo de la distribución, y luego redefinamos entropía:

# %% id="HRN9bcWH2QAp"
def probs(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return probs


# %% [markdown] id="j3-c29rk2QAp"
# Definimos un dataset de 6 elementos con dos atributos $X_1$ y $X_2$ con dos tipos de etiquetas:
#
# $False: -1$ y $True: 1$.
#
# Siendo las etiquetas resultantes del mismo:
#

# %% id="1ZR86tUm2QAq"
y = np.array([1, 1, 1, 1, 1, -1])

# %% [markdown] id="ekoGosuA2QAq"
# Mediante la función "probs" podemos calcular la probilidad de cada clase en este dataset:

# %% id="wxmrB-MD2QAq" outputId="35647efa-5b5f-4923-9618-0e66b2df1cb9"
probs(y)


# %% [markdown] id="33yftDzM2QAq"
# Esos resultados nos informan que hay un desbalance en el mismo (hay más probabilidad de una de las clases).
#
# Ahora obtengamos la entropía del dataset:

# %% id="W7h5J2tw2QAq"
def entropy(y):
    p = probs(y)
    return - np.sum(p * np.log2(p))


# %% id="86NJPaW32QAq" outputId="2fe50aee-2b4a-4a3a-e616-0eaf460680e7"
entropy(y)


# %% [markdown] id="NSfiW92n2QAq"
# ### Entropía Condicional
#
# Definamos entropía condicional:
#
# $$H(Y|X) = - \sum_{j=1}^v P(X = x_j) \sum_{i=1}^k P(Y = y_i | X = x_j) log_2 P(Y = y_i | X = x_j)$$
#
# Equivalentemente,
#
# $$H(Y|X) = \sum_{j=1}^v P(X = x_j) H(Y|X = x_j)$$
#
# Tomaremos $X$ binaria ($v=2$), por lo que la entropía condicional tendrá sólo dos términos.
#

# %% id="lp9Ql2oK2QAq"
def cond_entropy(y1, y2):
    size = y1.shape[0] + y2.shape[0]
    return y1.shape[0] / size * entropy(y1) + y2.shape[0] / size * entropy(y2)


# %% [markdown] id="HZgR2fYo2QAq"
# Esta función toma como argumento los dos subconjuntos de datos que se forman al considerar una de las variables como condición.

# %% [markdown] id="EdEAKVvc2QAs"
# Si analizámos la varibale $Y$ donde la $X_1$ es $True$, tenemos:

# %% id="ARHCGoCl2QAs"
y_X1true = np.array([1,1,1,1]) #X1 = True

# %% [markdown] id="8K0dWKdd2QAs"
# de igual manera, la varibale $Y$  donde la $X_1$ es $False$, tenemos:

# %% id="aZQhyoz12QAs"
y_X1false =  np.array([1,-1]) #X1 = False

# %% [markdown] id="TQt2bpar2QAs"
# Haciendo uso de la función "cond_entropy" podemos obtener la entropía condicional de la variable $Y$ dado $X_1$

# %% id="-d3NIWJX2QAs" outputId="972d7296-735a-491e-f70d-491b52d1aad2"
cond_entropy(y_X1true,y_X1false)  # x1

# %% [markdown] id="u2TpfXtC2QAs"
# De igual manera podemos calcular la entropía condicional dado la variable $X_2$

# %% id="TxIOGIAN2QAs" outputId="e6bfe319-0f78-4f65-d7a3-b6e3bcf757e3"
y_X2true = np.array([1,1,1]) #X2 = True
y_X2false = np.array([1,1,-1]) #X2 = False
cond_entropy(y_X2true, y_X2false)  # x2


# %% [markdown] id="nXodCaEC2QAs"
# ### Ganancia de Información
#
# La ganancia de información será simplemente la diferencia entre entropía y entropía condicional:

# %% id="4qqWqZZ52QAs"
def information_gain(y1, y2):
    y = np.concatenate((y1,y2))
    return entropy(y) - cond_entropy(y1,y2)


# %% [markdown] id="esvSX-9F2QAs"
# Podemos calcular la ganancia de información que implicaría dividir el dataset tomando como $X_1$ como nodo referencia. Para ello hacemos uso de la función "information_gain" que toma los dos subconjuntos de etiquetas que se formarían al tomar una u otra variable como referencia.

# %% [markdown] id="hrGWxPnQ2QAs"
# Para el caso de $X_1$:

# %% id="W0A7UQZ42QAt" outputId="6956b12f-f520-4b7f-9137-20ff9fe4c345"
information_gain(y_X1true, y_X1false)  # x1

# %% [markdown] id="J8gKbPhp2QAt"
# Para el caso de $X_2$:

# %% id="k_S47HMp2QAt" outputId="87191440-9587-454e-b27e-9f97171ddf8a"
information_gain(y_X2true, y_X2false)  # x2

# %% [markdown] id="yCfqafVc2QAt"
# ## Datos Sintéticos No Linealmente Separables
#
# Haremos algunos experimentos con datos generados sintéticamente. Estos datos serán no linealmente separables.
#
# Ejemplos típicos de datos no linealmente separables son los de tipo "OR", "AND" y "XOR". Usaremos datos de tipo "OR" para este ejemplo.
#
#

# %% id="VWbiAqX12QAu"
size = 200

# %% id="_JHgpvzL2QAv"
np.random.seed(0)
X = np.random.randn(size, 2)
y_true = np.logical_or(X[:, 0] > 0, X[:, 1] > 0)    # datos "OR"
#y_true = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)  # datos "XOR"
#y_true = np.logical_and(X[:, 0] > 0, X[:, 1] > 0)  # datos "AND"
y_true = y_true.astype(int)
y_true[y_true == 0] = -1

# %% id="_-umWIqL2QAv" outputId="e519ebc6-28d1-4cf7-f93c-fd0c0fa9c0aa"
plt.scatter(X[y_true==1, 0], X[y_true==1, 1], color="royalblue", label="1")
plt.scatter(X[y_true==-1, 0], X[y_true==-1, 1], color="red", label="-1")
plt.grid()
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.legend()
plt.show()

# %% [markdown] id="xkb4ONQ22QAv"
# ### División en Entrenamiento y Evaluación
#
# Separemos la mitad para entrenamiento y la otra para evaluación.

# %% id="EDd_9DfQ2QAv"
train_size = 100
test_size = size - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y_true[:train_size], y_true[train_size:]

# %% id="WmXdk9D22QAv" outputId="4f77df13-0bb1-40a0-e5f9-6e65f77dfe61"
X_train.shape, X_test.shape

# %% [markdown] id="eOdUQ7nO2QAv"
# ### Clasificación Lineal
#
# Veamos qué tan mal anda un clasificador lineal sobre estos datos.

# %% id="6xp5rgil2QAv"
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train);

# %% id="cBYfK5Nb2QAv" outputId="c4e47a9e-380b-496b-b15b-0c020ee9e7f4"
!pip3 install utils
from utils import plot_decision_boundary
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plot_decision_boundary(lambda x: model.predict(x), X, y_true)

# %% [markdown] id="Em9ICma_2QAv"
# Calculemos la calidad de la predicción en entrenamiento y evaluación:

# %% id="Yj2vNEfX2QAv"
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# %% id="iVZr861l2QAw" outputId="1fa9428b-2a03-4d0f-a32f-594192175360"
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy: {train_acc:0.2}')
print(f'Test accuracy: {test_acc:0.2}')

# %% [markdown] id="3ZwLnqdQ2QAw"
# ### Nota al Margen: Induciendo Separabilidad Lineal
#
# Muchas veces se pueden convertir datos no linealmente separables en datos separables (o casi) mediante la introducción de nuevos atributos que combinan los atributos existentes.
# Un ejemplo de estos son los atributos polinomiales.
#
# Aquí lo haremos con datos "OR", pero la diferencia es mucho más notable con datos de tipo "XOR".

# %% id="XIrGswHv2QAw"
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pre = PolynomialFeatures(
    degree=2,
    interaction_only=True,  # para usar solo x0*x1, no x0*x0 ni x1*x1
    include_bias=False)
X_train2 = pre.fit_transform(X_train)

# %% id="yDdsHvIg2QAw" outputId="8e01221d-a6a7-4227-c8d7-6558926989c7"
X_train.shape, X_train2.shape  # se agregó el feature x0*x1

# %% [markdown] id="djeC6yD02QAw"
# Grafiquemos:

# %% id="-wYYzVfP2QAw" outputId="7b672485-9d62-4135-bb6b-d70904c9feee"
plt.scatter(X_train2[y_train==1, 1], X_train2[y_train==1, 2], color="dodgerblue", edgecolors='k', label="1")
plt.scatter(X_train2[y_train==-1, 1], X_train2[y_train==-1, 2], color="tomato", edgecolors='k', label="-1")
plt.grid()
plt.legend()
plt.show()

# %% id="XwJrY8bJ2QAx"
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    LogisticRegression()
)
model.fit(X_train, y_train);

# %% id="NeWByl8d2QAx"
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# %% id="Fw1oX-1I2QAx" outputId="70d65169-0b33-4b09-927a-0776d9f79de5"
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy: {train_acc:0.2}')
print(f'Test accuracy: {test_acc:0.2}')

# %% id="LsHMsg8Y2QAx" outputId="84967ece-6ee9-4003-fc3e-0834b8b9e0a0"
from utils import plot_decision_boundary

plot_decision_boundary(lambda x: model.predict(x), X, y_true)
plt.xlabel("X[0]")
plt.ylabel("X[1]")

# %% [markdown] id="JuK4f5Fp2QAx"
# ### Entropía y Valores Reales
#
# Calculemos la entropía inicial, y veamos cómo condicionar la entropía sobre variables reales (i.e. no categóricas)

# %% id="nhFMCvV12QAx" outputId="e2e4c8f4-f54e-42d9-d073-1d89a1972c98"
y_train

# %% [markdown] id="jpJqKulD2QAy"
# Podemos calcular la probabilidad de cada clase:

# %% id="5k78PHVh2QAy" outputId="436e344b-29a3-4219-d6dc-aab967d3b545"
probs(y_train)

# %% [markdown] id="wBvOnlo52QAy"
# y la entropía:

# %% id="i6f4Hkgq2QAy" outputId="ffd9790e-8437-4440-bcc7-4190c81ce546"
entropy(y_train)


# %% [markdown] id="EY9aIXtc2QAz"
# Para hacer una división sobre una variable real usaremos un valor "threshold" (umbral):

# %% id="TDo1k87b2QAz"
def split(X, y, i, threshold):
    y1 = y[X[:, i] > threshold]
    y2 = y[X[:, i] <= threshold]
    return y1, y2


# %% [markdown] id="M9V89Mrt2QAz"
# definimos la "split" que toma como argumento el dataset con sus etiquetas, el indice de la variable que usaremos para dividir el dataset y el umbral para realizarlo y nos devuelve dos subconjuntos de este dataset.

# %% [markdown] id="-gHefI6g2QAz"
# Si utilizamos la variable $X_1$ con un umbral de $0.00$ obtenemos:

# %% id="Ram2bqqJ2QAz"
y1, y2 = split(X_train, y_train, 0, 0.00)

# %% id="AuClPUXM2QAz" outputId="26200b25-39df-4213-ccdb-b7ee8bb5e181"
y1,y2

# %% [markdown] id="s0sx6LJY2QAz"
# Siendo la entropía de cada uno de ellos:

# %% id="dvf2V3u82QAz" outputId="935b06fe-cc7b-47cb-c476-9bae08f52ed5"
entropy(y1), entropy(y2)

# %% id="NwIM2wRK2QAz" outputId="24815423-c644-4f47-906e-6a02644952d5"
cond_entropy(y1, y2)

# %% id="qs2Ktp602QA0" outputId="97e21561-13fc-4ebb-ffa0-4bc9a3c8949c"
information_gain(y1,y2)

# %% [markdown] id="QgEWOSwB2QA0"
# ### Buscando la Mejor División
#
# Ilustraremos un paso en la construcción del árbol de decisión.
#
# Probemos muchos threshold para ambas variables y seleccionemos la mejor división.
#
# En este caso buscaremos en una grilla uniforme de valores, pero existen técnicas mejores.

# %% id="xuCNCuEH2QA0" outputId="5b732bca-71de-4133-a591-64b38f37f583"
np.linspace(-2.5, 2.5, 11)

# %% id="_TGtDGc72QA0" outputId="635ab305-88dc-4667-ebea-8a6bd1bfb3a9"
best_ig = 0

for i in [0, 1]:
    for threshold in np.linspace(-2.5, 2.5, 11):
        y1, y2 = split(X_train, y_train, i, threshold)
        ig = information_gain(y1, y2)
        print(f'i={i}\tthreshold={threshold:+00.2f}\tig={ig:.2f}')

        if ig >= best_ig:
            best_ig = ig
            best_feature = i
            best_threshold = threshold

print('Mejor división:')
print(f'feature={best_feature}, threshold={best_threshold}, ig={best_ig:00.2f}')


# %% [markdown] id="P-nhTeGW2QA0"
# Dividamos los datos de acuerdo a esta frontera:

# %% id="ofQv5yB-2QA0" outputId="9c64b29a-96af-4d32-b3c6-e51a01d37b3f"
best_feature, best_threshold

# %% [markdown] id="A-7SUpWC2QA1"
# Podemos ver los datos en el gráfico

# %% id="BTPncvwc2QA1" outputId="aecad9e4-8b1d-4f18-ce21-1e82e29ae8d2"
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color="royalblue", label="1")
plt.scatter(X_train[y_train==-1, 0], X_train[y_train==-1, 1], color="red", label="-1")

plt.axhline(y=0, xmin=-3, xmax=3, color='green', linestyle='-.', linewidth=2)

plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.grid()
plt.legend()
plt.show()

# %% id="6g3t9dvo2QA1"
y1, y2 = split(X_train, y_train, best_feature, best_threshold)

# %% [markdown] id="F7ioaEd52QA1"
# Con esta división, la entropía baja considerablemente:

# %% id="hchFCjFw2QA1" outputId="a8a68af5-6340-462d-a067-c2dfae78bfaf"
entropy(y_train)

# %% id="VlTY_o3_2QA1" outputId="260f0b1c-9bf3-414b-aeb2-13b1df356459"
cond_entropy(y1, y2)

# %% [markdown] id="lSR-7zN62QA2"
# ## Árbol de Decisión con Scikit-learn
#
# Aprendamos un árbol de decisión usando scikit-learn. Para ello usaremos la clase [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html):
#

# %% id="x6LDM0cz2QA2" outputId="015db4ea-8342-47b7-a98e-251ae0636847"
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

# %% [markdown] id="bdPGcGAC2QA2"
# Ahora predecimos y evaluamos:

# %% id="Jcd721TC2QA2"
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# %% id="dKFxBKEs2QA2" outputId="d7c6b86b-8f6e-4438-838f-893729544e3f"
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy: {train_acc:0.2}')
print(f'Test accuracy: {test_acc:0.2}')

# %% [markdown] id="kePbsJlf2QA2"
# Dibujamos la frontera de decisión:

# %% id="SzIV9OXQ2QA2" outputId="b42f2673-3f9d-4d13-9060-921f1ba0cd64"
from utils import plot_decision_boundary

plot_decision_boundary(lambda x: clf.predict(x), X_train, y_train)
plt.xlabel("X[0]")
plt.ylabel("X[1]")

# %% [markdown] id="HHdYsk492QA2"
# También podemos inspeccionar el árbol:

# %% id="l4rihIdo2QA3" outputId="66b7043c-62f3-41a0-b168-dc014b2979e6"
from sklearn.tree import plot_tree

plot_tree(clf,filled=True);

# %% id="ad25ZpIF2QA4" outputId="2d69e939-60eb-4ec2-d84c-7fe5351d133b"
entropy(y_train)

# %% [markdown] id="GoDU-yNK2QA4"
# ## Ejercicios
#
# 1. Probar todos los experimentos con un dataset de tipo "XOR". ¿Qué sucede al decidir la división en el primer nivel del árbol?

# %% [markdown] id="WYEA_sTD2QA4"
# ## Referencias
#
# Scikit-learn:
#
# - [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
# - [User Guide: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
# - [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
#
