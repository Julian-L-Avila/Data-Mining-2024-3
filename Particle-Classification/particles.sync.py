V
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

# %% [markdown] id="ML5YoNrZS_Gc"
# # Parcial 2, métodos de clasificación con datos del LHC
#
#

# %% id="scz89F7WS_Ge"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.svm import SVC, LinearSVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# %% id="-V-0pnD9S_Gf"
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)
rc('text', usetex=False)

# %% [markdown] id="eTvCct6rS_Gg"
# ### Leemos el .csv y nombramos las columnas

# %% id="aMESRMOuS_Gg" outputId="6a0db168-8f9e-4162-feb1-32e76a7a61e7"
df = pd.read_csv('TrainingValidationData.csv', delimiter=',', names=list(['P'+str(i) for i in range(73)]) )
print(df.columns)

# %% id="zUXOw8hmS_Gh" outputId="6f7a78bc-3b36-46a5-ca1e-ca26a2741105"
df.head()

# %% [markdown]
# ### El dataset trae en P0 6 caracteristicas más, entonces las dividimos.

# %% id="qM0B1tcoS_Gh"
new = df['P0'].str.split(';',expand=True)

new.columns = ['numID', 'processID', 'weight', 'MET', 'METphi', 'Type_1']

# %% id="WTm89vPhS_Gi" outputId="843fc4f4-0b86-4752-ea94-10af41dad00e"
new.head()

# %% [markdown]
# ### Se unen al dataset principal

# %% id="9awdDNoMS_Gj"
df = df.join(new, how='outer') #join them side to side

# %% id="Kv8H5LlMS_Gk" outputId="ff6ee078-39ed-4377-da5c-81dc62aaa9ba"
df.head()

# %% [markdown] id="U-rz8L4VS_Gl"
# ### Convertimos Type_1 en 14 columnas

# %% id="VvWtTR32S_Gl" outputId="27803b0c-8920-41b2-8678-8a8a41ac3711"
for i in range(4,53,4):

    new = df['P'+str(i)].str.split(';',expand=True) 
    
    df['P'+str(i)] = new[0]
    
    df['Type_'+str(int(i/4+1))] = new[1]
    
print(df.columns)

# %% id="qcOp2M0jS_Gm"
df = df.drop('P0', axis=1)

# %% id="mkPK8TmfS_Gm" outputId="984713ad-297a-49bf-a8f0-194355b0def4"
df.columns.values

# %% [markdown]
# ### Ordenamos

# %% id="St2vlWUVS_Gn"
#just re-ordering

cols = ['numID', 'processID', 'weight',
       'MET', 'METphi', 'Type_1', 'P1', 'P2', 'P3', 'P4',  'Type_2', 'P5', 'P6', 'P7', 'P8', 'Type_3', 'P9', 'P10', 'P11',
       'P12',  'Type_4', 'P13', 'P14', 'P15', 'P16', 'Type_5','P17', 'P18', 'P19', 'P20',
       'Type_6','P21', 'P22', 'P23', 'P24', 'Type_7','P25', 'P26', 'P27', 'P28', 'Type_8','P29',
       'P30', 'P31', 'P32', 'Type_9', 'P33', 'P34', 'P35', 'P36', 'Type_10','P37', 'P38',
       'P39', 'P40', 'Type_11', 'P41', 'P42', 'P43', 'P44', 'Type_12', 'P45', 'P46', 'P47',
       'P48', 'Type_13','P49', 'P50', 'P51', 'P52']

# %% id="YPw8KyOPS_Gn"
X = df[cols].drop(['numID', 'processID', 'weight'], axis = 1)

# %% id="iz7CfTQcS_Gn" outputId="ad529c19-dbea-48a9-de13-f9d59e60e1f0"
len(cols)

# %% id="sQQbpVLmS_Go" outputId="f72074c1-5355-409a-cb5a-bc1e622ab6bd"
X.head() 

# %% id="fcN5KgfQS_Go" outputId="c6af399b-389f-471f-a5eb-2cdedddb8975"
X.describe() 

# %% [markdown] id="guwdTWo4S_Go"
# Some columns that should be numerical are of type "object"

# %% id="3Wdvnm3tS_Gp" outputId="855a4dbc-1560-426b-e601-6ea1e176513b"
X.columns[X.dtypes == object]

# %% [markdown] id="Shj0F3e3S_Gp"
# ### formateamos al correcto

# %% id="-p-FR80cS_Gp"
for el in ['MET', 'METphi', 'P4', 'P8', 'P12',
        'P16',  'P20', 'P24',  'P28',
    'P32', 'P36', 'P40', 'P44',
      'P48', 'P52']:
    X[el] = X[el].astype('float64')

# %% id="kce9nvvVS_Gp" outputId="77f7d7f2-9a21-4b29-8036-93f7ec085ed8"
X.dtypes

# %% [markdown] id="9djrPJsFS_Gq"
# ### Selecionamos solamente 5000 datos para que sea manipulable

# %% id="4OoIekILS_Gq"
np.random.seed(10)

sel = np.random.choice(df.shape[0], 5000)

features = X.iloc[sel,:]

# %% id="NT6yb-_BS_Gq" outputId="86adc8c3-1759-451b-870f-d9ed9a4c7df2"
features.shape

# %% id="kHFeOI0ZS_Gr" outputId="9a40f6a6-a114-4b58-f09e-1e21141c61f1"
features.columns

# %% [markdown] id="rCuVEnv0S_Gr"
# ### reseteamos indices

# %% id="6shhXkSIS_Gr"
features.reset_index(drop=True, inplace=True)

# %% id="6rKA1ucAS_Gs" outputId="9d3d7e7c-c1da-4e9d-bcbc-c2abda047d8d"
features.head()

# %% [markdown] id="_2FRSAe9S_Gs"
# ### Exportamos las caracteristicas

# %% id="QlRvXY0pS_Gs"
features.to_csv('ParticleID_features.csv', index_label= 'ID')

# %% [markdown] id="vuOtLqvOS_Gs"
# #### Seleccionamos los target

# %% id="BHaB0IvjS_Gt"
y = df.processID[sel].values # values makes it an array

# %% id="cTqvxDaNS_Gt" outputId="81c85cb5-12cc-4d9b-a118-2579b7aaced3"
y

# %% [markdown] id="irjKppmrS_Gt"
# ### Los exportamos

# %% id="6QQy_LVcS_Gt"
np.savetxt('ParticleID_labels.txt', y, fmt = '%s')

# %% [markdown] id="UGSpyK91S_Gu"
# ## Iniciamos el analisis:

# %% id="Xd7WnozpS_Gu"
features = pd.read_csv('ParticleID_features.csv', index_col='ID')

# %% id="dHs0SGxUS_Gu" outputId="32a6103a-31d2-413d-f3ab-9eb7a10c2c9f"
features.head()

# %% id="KAl-0b3ES_Gv" outputId="3b896999-0a88-4101-e869-42ce0c14d3c8"
features.shape

# %% id="82IFUA9YS_Gv"
y = np.genfromtxt('ParticleID_labels.txt', dtype = str)

# %% id="A-Y9Mg5XS_Gv" outputId="0518e8db-5e1d-44c4-dbb5-1d500511eff7"
y

# %% [markdown] id="1EQH55WhS_Gv"
# ### convertimos el target en 0 y 1

# %% id="lzB-AaSrS_Gv"
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() #turns categorical into 1 ... N

# %% id="OQSG8hh7S_Gw" outputId="14e4c4ed-585a-405e-b59a-ab66f9d3f670"
y

# %% id="vMpkms7qS_Gw"
y = le.fit_transform(y)

# %% id="7DwLiw0mS_Gw" outputId="8ceab38c-aa77-4769-eafb-7755815c574f"
y 

# %% [markdown]
# ### invertimos los numeros por comodidad

# %% id="t7WQyDGwS_Gx"
target = np.abs(y - 1)

# %% id="Wsr1pOy8S_Gx" outputId="a01e48ff-95e6-432a-e7a5-11e6a964f356"
target

# %% [markdown] id="SoQXel9US_Gy"
# ### Solo consideramos las primeras 16 columnas (los primeros cuatro productos) para que tengamos problemas de imputación/manipulación limitados.
#
# Tenemos que elegir entre mantener más características, pero tener un problema de imputación/datos faltantes más grave, o mantener menos características, pero lidiar con un problema de imputación más simple. Elegimos la segunda opción.

# %% id="p_L-a_R5S_Gy"
features_lim = features[['MET', 'METphi', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11',
       'P12',  'P13', 'P14', 'P15', 'P16']]

features_lim.describe()
# %% id="IV1hb_t2S_Gy" outputId="f3601d2c-6449-4fe3-fa7c-afc5d0849f36"
features_lim.head()

# %% id="1-Lf0IDwS_Gz" outputId="47579686-b9af-4d2a-8354-4fe630c86c45"
features_lim.describe() 

# %% [markdown] id="bkkYvd7PS_Gz"
# ¡Aún quedan algunas columnas de características con longitudes diferentes! Esto significa que puede haber valores NaN. Reemplácelos por 0 por el momento.

# %% id="WZYyTYhrS_Gz"
features_lim = features_lim.fillna(0) #Fill with 0 everywhere there is a NaN

# %% [markdown]
# Nota: esta es la opción más simple pero la peor posible: imputar un valor constante distorsiona el modelo :D Un paso más sería ingresar la media o la mediana para cada columna. Sin embargo, debido a que solo una cantidad limitada de instancias tienen datos faltantes, la elección de la estrategia de imputación no importa demasiado.

# %% id="vbnrvY0rS_G0" outputId="87922e80-d595-4dc9-da1b-34927e2ef184"
features_lim.describe()

# %% [markdown]
# # Usando metodos de clasificación:

# %% [markdown]
# ## Se prepara dataset:

# %%
import pandas as pd
import numpy as np

df = pd.read_csv("ParticleID_features.csv")

df_lim = df[['MET', 'METphi', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11',
       'P12',  'P13', 'P14', 'P15', 'P16']]

df_lim.head()

# %%
df_lim.describe()

# %%
df_lim = df_lim.fillna(0) #Fill with 0 everywhere there is a NaN
df_lim.describe()

# %%
y = np.genfromtxt('ParticleID_labels.txt', dtype = str)

y


# %% [markdown]
# ### Volvemos los target como valores numéricos.

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() #turns categorical into 1 ... N

y = le.fit_transform(y)

target = np.abs(y - 1)

target

# %% [markdown]
# ### Competimos contra una clasificacion donde todo dato que llegue lo clasificamos como 4top, de una precisión de 0.8378

# %%
np.sum(target)/len(target) #distribution 

# %% [markdown]
# ## SVM:

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics


df_svm = df_lim

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(df_svm, target, test_size=0.2, random_state=42)


X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)       


modelo_svm = SVC(kernel='rbf', C=1.0)  
modelo_svm.fit(X_train, y_train)


y_pred = modelo_svm.predict(X_test)


print("Exactitud:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Matriz de confusión:\n", metrics.confusion_matrix(y_test, y_pred))

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

piped_model = make_pipeline(StandardScaler(), SVC()) 

piped_model.get_params() 

# %%

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


parameters = {'svc__kernel':['poly', 'rbf'], \
              'svc__gamma':[0.00001,'scale', 0.01, 0.1], 'svc__C':[0.1, 1.0, 10.0, 100.0, 1000], \
              'svc__degree': [2, 4, 8]}

model = GridSearchCV(piped_model, parameters, cv = StratifiedKFold(n_splits=5, shuffle=True), \
                     verbose = 2, n_jobs = 4, return_train_score=True)

model.fit(X_train, y_train)

print('Best params, best score:', "{:.4f}".format(model.best_score_), \
      model.best_params_)

# %% [markdown]
# ## Tree Desicion:

# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt


df_tree = df_lim


X_train, X_test, y_train, y_test = train_test_split(df_tree, target, test_size=0.2, random_state=42)


modelo_tree = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=3,     
    min_samples_split=10
)
modelo_tree.fit(X_train, y_train)


y_pred = modelo_tree.predict(X_test)


print("Exactitud:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Matriz de confusión:\n", metrics.confusion_matrix(y_test, y_pred))


plt.figure(figsize=(15,10))
plot_tree(
    modelo_tree,
    feature_names=df_tree.columns[:18].tolist(), 
    class_names=[str(clase) for clase in modelo_tree.classes_], 
    filled=True,
    rounded=True
)
plt.show()

# %% [markdown]
# ### Conocer la importancia de las caracteristicas:

# %%
importancias = modelo_tree.feature_importances_
for i, importancia in enumerate(importancias):
    print(f"Característica {i+1}: {importancia:.4f}")

# %% [markdown]
# ### buscando los mejores hiperparametros

# %%
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


param_dist = { 
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced']
}


random_search = RandomizedSearchCV(
    DecisionTreeClassifier(),
    param_distributions=param_dist,
    n_iter=100,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)


print("=== Mejores hiperparámetros (Randomized) ===")
print(random_search.best_params_)
print("Precisión (CV):", random_search.best_score_)


param_grid_refined = {
    'max_depth': [3, 4, 5],  
    'min_samples_split': [8, 10, 12],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [8, 10, 12]  
}

grid = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid_refined,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)


print("\n=== Mejores hiperparámetros (Grid) ===")
print(grid.best_params_)
print("Precisión (CV):", grid.best_score_)


mejor_modelo = grid.best_estimator_  
y_pred = mejor_modelo.predict(X_test)
print("\nExactitud en prueba:", metrics.accuracy_score(y_test, y_pred))

# %% [markdown]
# ### Eliminemos las caracteristicas con menor peso

# %%
df_tree

# %%
features_relevantes = [0, 7, 11, 15] 
df_tree = df_tree.iloc[:, features_relevantes]
df_tree

# %%
X_train, X_test, y_train, y_test = train_test_split(df_tree, target, test_size=0.2, random_state=42)

param_dist = {  
    'max_depth': [3, 5, 7, 10, 15, 20],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}


random_search = RandomizedSearchCV(
    DecisionTreeClassifier(),
    param_distributions=param_dist,
    n_iter=100,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)


print("=== Mejores hiperparámetros (Randomized) ===")
print(random_search.best_params_)
print("Precisión (CV):", random_search.best_score_)


param_grid_refined = {
    'max_depth': [random_search.best_params_['max_depth'] - 2, 
                  random_search.best_params_['max_depth'], 
                  random_search.best_params_['max_depth'] + 2],
    'min_samples_split': [random_search.best_params_['min_samples_split'] // 2,
                          random_search.best_params_['min_samples_split'],
                          random_search.best_params_['min_samples_split'] * 2],
    
}

grid = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid_refined,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)


print("\n=== Mejores hiperparámetros (Grid) ===")
print(grid.best_params_)
print("Precisión (CV):", grid.best_score_)


mejor_modelo = grid.best_estimator_  
y_pred = mejor_modelo.predict(X_test)
print("\nExactitud en prueba:", metrics.accuracy_score(y_test, y_pred))

# %%
from sklearn.tree import plot_tree
plt.figure(figsize=(10,6))
plot_tree(mejor_modelo, feature_names=df_tree.columns.tolist(), filled=True)
plt.show()

# %% [markdown]
# ## Logistic Regresion:

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


df_log = df_lim

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_log)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Crear y entrenar el modelo
modelo_lr = LogisticRegression(
    penalty='l2',         
    C=1.0,                
    solver='liblinear',   
    max_iter=1000         
)
modelo_lr.fit(X_train, y_train)


y_pred = modelo_lr.predict(X_test)


print("Exactitud:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Matriz de confusión:\n", metrics.confusion_matrix(y_test, y_pred))


coeficientes = pd.DataFrame({
    'Característica': df_log.columns[:18],
    'Coeficiente': modelo_lr.coef_[0]
})
print("\nCoeficientes:\n", coeficientes.sort_values(by='Coeficiente', ascending=False))

# %%
modelo_lr = LogisticRegression(
    penalty='l2',       
    C=1.0,              
    solver='liblinear', 
    max_iter=1000,
    class_weight={0: 5, 1: 1}
)
modelo_lr.fit(X_train, y_train)


y_pred = modelo_lr.predict(X_test)


print("Exactitud:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Matriz de confusión:\n", metrics.confusion_matrix(y_test, y_pred))


coeficientes = pd.DataFrame({
    'Característica': df_log.columns[:18],
    'Coeficiente': modelo_lr.coef_[0]
})
print("\nCoeficientes:\n", coeficientes.sort_values(by='Coeficiente', ascending=False))

# %% [markdown]
# ### probando diferentes hiperparametros

# %%
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2','l1'],
    'class_weight': [None, {0: 3, 1: 1}, 'balanced'],
    'solver': ['saga', 'liblinear'],
    'max_iter': [1000, 500, 10000, 100000]
}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1')


grid.fit(X_train, y_train)


print("\n=== Mejores hiperparámetros (Grid) ===")
print(grid.best_params_)
print("Precisión (CV):", grid.best_score_)


mejor_modelo = grid.best_estimator_  
y_pred = mejor_modelo.predict(X_test)
print("\nExactitud en prueba:", metrics.accuracy_score(y_test, y_pred))

# %%
modelo_lr = LogisticRegression(
    penalty='l2',         
    C=1.0,           
    solver='saga',   
    max_iter=1000,
    class_weight=None
)
modelo_lr.fit(X_train, y_train)


y_pred = modelo_lr.predict(X_test)

# Metricas
print("Exactitud:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Matriz de confusión:\n", metrics.confusion_matrix(y_test, y_pred))


coeficientes = pd.DataFrame({
    'Característica': df_log.columns[:18],
    'Coeficiente': modelo_lr.coef_[0]
})
print("\nCoeficientes:\n", coeficientes.sort_values(by='Coeficiente', ascending=False))

# %% [markdown]
# ### eliminando caracteristicas

# %%
features_relevantes = [0, 7, 11, 15, 3,5,2] 
df_log = df_log.iloc[:, features_relevantes]
df_log

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_log)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# %%
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2','l1'],
    'class_weight': [None, {0: 3, 1: 1}, 'balanced'],
    'solver': ['saga', 'liblinear'],
    'max_iter': [1000, 500, 10000, 100000]
}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1')


grid.fit(X_train, y_train)


print("\n=== Mejores hiperparámetros (Grid) ===")
print(grid.best_params_)
print("Precisión (CV):", grid.best_score_)


mejor_modelo = grid.best_estimator_  
y_pred = mejor_modelo.predict(X_test)
print("\nExactitud en prueba:", metrics.accuracy_score(y_test, y_pred))

# %% [markdown]
# ## KNN: 

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

df_knn = df_lim


X_train, X_test, y_train, y_test = train_test_split(
    df_knn, 
    target, 
    test_size=0.2, 
    stratify=target, 
    random_state=42
)

# pipeline (escalado + modelo)
knn_model = make_pipeline(
    StandardScaler(),  
    KNeighborsClassifier(n_neighbors=5)  
)


knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)


print("=== KNN ===")
print(f"Exactitud: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 10] 
}

grid = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Mejores parámetros:", grid.best_params_)
print("Mejor exactitud (CV):", grid.best_score_)

# %% [markdown]
# ## Bayes:

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler  

df_bayes = df_lim


X_train, X_test, y_train, y_test = train_test_split(
    df_bayes, 
    target, 
    test_size=0.2, 
    stratify=target, 
    random_state=42
)

# Pipeline (escalado opcional + modelo)
nb_model = make_pipeline(
    StandardScaler(), 
    GaussianNB()      
)


nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)


print("=== Naive Bayes ===")
print(f"Exactitud: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'gaussiannb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6] 
}

grid = GridSearchCV(nb_model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Mejor parámetro:", grid.best_params_)
print("Mejor exactitud (CV):", grid.best_score_)

# %% [markdown]
# ## Ensambles:

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

df_ens = df_lim


X_train, X_test, y_train, y_test = train_test_split(
    df_ens, 
    target, 
    test_size=0.2, 
    stratify=target,  
    random_state=42
)

# 3. Modelo Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("=== Random Forest ===")
print("Exactitud:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_rf))

# Importancia de características (RF)
importancias = rf_model.feature_importances_
indices = np.argsort(importancias)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Importancia de características (Random Forest)")
plt.bar(range(X_train.shape[1]), importancias[indices], align="center")
plt.xticks(range(X_train.shape[1]), df_ens.columns[indices], rotation=90)
plt.show()

# 4. Modelo Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("\n=== Gradient Boosting ===")
print("Exactitud:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_gb))

# Validación cruzada (opcional)
print("\nValidación cruzada RF (5 folds):")
scores = cross_val_score(rf_model, df_ens, target, cv=5, scoring='accuracy')
print("Promedio:", scores.mean(), "±", scores.std())

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt


df_ens = df_lim


X_train, X_test, y_train, y_test = train_test_split(
    df_ens, 
    target, 
    test_size=0.2, 
    stratify=target,  
    random_state=42
)

# 2. Manejo de desbalance: Calcular pesos para Gradient Boosting
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)


rf_initial = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_initial.fit(X_train, y_train)

# Obtener características relevantes (umbral > 0.01)
importancias = rf_initial.feature_importances_
umbral = 0.01  # Ajusta según tus resultados
seleccion = importancias > umbral
X_train_sel = X_train.loc[:, seleccion]
X_test_sel = X_test.loc[:, seleccion]

# 4. Optimización de hiperparámetros con GridSearchCV
# ==============================================================================
# Random Forest
# ==============================================================================
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train_sel, y_train)

# Mejor modelo RF
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_sel)

# ==============================================================================
# Gradient Boosting
# ==============================================================================
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [3, 5, 7]
}

grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid_gb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_gb.fit(X_train_sel, y_train, sample_weight=sample_weights)

# Mejor modelo GB
best_gb = grid_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test_sel)

# 5. Evaluación de resultados
# ==============================================================================
# Resultados Random Forest
print("=== Mejor Random Forest ===")
print("Parámetros:", grid_rf.best_params_)
print("Exactitud (test):", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_rf))

# Resultados Gradient Boosting
print("\n=== Mejor Gradient Boosting ===")
print("Parámetros:", grid_gb.best_params_)
print("Exactitud (test):", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_gb))

# 6. Importancia de características (post-selección)
plt.figure(figsize=(10, 6))
plt.bar(range(X_train_sel.shape[1]), best_rf.feature_importances_, align="center")
plt.xticks(range(X_train_sel.shape[1]), X_train_sel.columns, rotation=90)
plt.title("Importancia de características (Mejor Random Forest)")
plt.show()

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight


df_ens = df_lim

X_train, X_test, y_train, y_test = train_test_split(
    df_ens, 
    target, 
    test_size=0.2, 
    stratify=target,  
    random_state=42
)

# 2. Calcular pesos para Gradient Boosting (manejo de desbalance)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 3. Selección de características (eliminar <1% importancia)
# Entrenar modelo inicial para obtener importancias
rf_initial = RandomForestClassifier(
    class_weight='balanced', 
    random_state=42
).fit(X_train, y_train)

umbral = 0.01  # Ajustar según necesidad
seleccion = rf_initial.feature_importances_ > umbral
X_train_sel = X_train.loc[:, seleccion]
X_test_sel = X_test.loc[:, seleccion]

# 4. Optimización con foco en recall de clase 1
# ==============================================================================
# Random Forest (énfasis en clase minoritaria)
# ==============================================================================
param_grid_rf = {
    'n_estimators': [200, 300],         
    'max_depth': [10, None],          
    'min_samples_split': [5, 10],     
    'class_weight': [{0:1, 1:5}, 'balanced'] 
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=5,
    scoring='recall',  # Priorizar recall clase 1
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train_sel, y_train)

# ==============================================================================
# Gradient Boosting (ponderación manual)
# ==============================================================================
param_grid_gb = {
    'n_estimators': [200, 300],      
    'learning_rate': [0.1, 0.05],    
    'max_depth': [3, 5],             
    'subsample': [0.8, 1.0]          
}

grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid_gb,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=1
)
grid_gb.fit(X_train_sel, y_train, sample_weight=sample_weights)  # Ponderación

# 5. Evaluación final
# ==============================================================================
# Mejor Random Forest
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_sel)

print("=== Mejor Random Forest ===")
print(f"Parámetros: {grid_rf.best_params_}")
print("Exactitud:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_rf))

# Mejor Gradient Boosting
best_gb = grid_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test_sel)

print("\n=== Mejor Gradient Boosting ===")
print(f"Parámetros: {grid_gb.best_params_}")
print("Exactitud:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_gb))

# 6. Importancia de características (post optimización)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(X_train_sel.shape[1]), best_rf.feature_importances_, align="center")
plt.xticks(range(X_train_sel.shape[1]), X_train_sel.columns, rotation=90)
plt.title("Importancia de características (Modelo Optimizado)")
plt.show()

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Cargar datos
df_ens = df_lim
X_train, X_test, y_train, y_test = train_test_split(df_ens, target, test_size=0.2, stratify=target, random_state=42)

# Manejo de desbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Selección de características con Random Forest
rf_initial = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_initial.fit(X_train, y_train)
importancias = rf_initial.feature_importances_
umbral = 0.01
seleccion = importancias > umbral
X_train_sel = X_train.loc[:, seleccion]
X_test_sel = X_test.loc[:, seleccion]

# Definición de modelos base
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
svm = SVC(probability=True, random_state=42)

# Bagging
bagging_dt = BaggingClassifier(estimator=dt, n_estimators=100, random_state=42)
bagging_dt.fit(X_train_sel, y_train)
y_pred_bagging = bagging_dt.predict(X_test_sel)

# Stacking
stacking = StackingClassifier(
    estimators=[('knn', knn), ('nb', nb), ('dt', dt), ('svm', svm)],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)
stacking.fit(X_train_sel, y_train)
y_pred_stacking = stacking.predict(X_test_sel)

# Voting
voting = VotingClassifier(
    estimators=[('knn', knn), ('nb', nb), ('dt', dt), ('svm', svm)],
    voting='soft'
)
voting.fit(X_train_sel, y_train)
y_pred_voting = voting.predict(X_test_sel)

# Evaluación
models = {
    'Bagging Decision Tree': y_pred_bagging,
    'Stacking': y_pred_stacking,
    'Voting': y_pred_voting
}

for name, y_pred in models.items():
    print(f"\n=== {name} ===")
    print("Exactitud:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

df_svm = df_lim

# Select only P14 and P10 features
X = df_svm[['P14', 'P10']].values
y = target # Assuming target is a Pandas Series

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
modelo_svm = SVC(kernel='rbf', C=1.0)
modelo_svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = modelo_svm.predict(X_test)
print("Exactitud:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Matriz de confusión:\n", metrics.confusion_matrix(y_test, y_pred))

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('P14 (scaled)')
    plt.ylabel('P10 (scaled)')
    plt.title('SVM Decision Boundary')
    plt.show()

# Visualize the hyperplanes
plot_decision_boundary(modelo_svm, X_train, y_train)

# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

df_svm = df_lim

# Select only P14 and P10 features
X = df_svm[['P14', 'P6']].values
y = target # Assuming target is a Pandas Series

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
modelo_svm = SVC(kernel='rbf', C=1.0)
modelo_svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = modelo_svm.predict(X_test)
print("Exactitud:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Matriz de confusión:\n", metrics.confusion_matrix(y_test, y_pred))

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('P14 (scaled)')
    plt.ylabel('P6 (scaled)')
    plt.title('SVM Decision Boundary')
    plt.show()

# Visualize the hyperplanes
plot_decision_boundary(modelo_svm, X_train, y_train)

# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_lim['target'] = target

sns.pairplot(df_lim, hue='target', palette='Set1', markers=['o', 's'])
plt.suptitle("Pairplot of df_lim Dataset Features (Binary Classification)", y=1.02)
plt.show()
