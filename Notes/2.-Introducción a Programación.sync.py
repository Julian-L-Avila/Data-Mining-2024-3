# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2.-Introducción a la Programación con Python

# %% [markdown]
# ## 2.2.- Operadores Aritméticos

# %% [markdown]
# ### 2.2.1.- Variables y asignaciones

# %% [markdown]
# Ya has usado iPython de calculadora con expresiones como sumas

# %%

# %% [markdown]
# Creemos una variable w que valga 39

# %%
w = 39

# %%

# %% [markdown]
# Es útil realizar estas declaraciones, ya que las declaraciones permiten almacenar valores que pueden ser útiles posteriormente. Por ejemplo:

# %%

# %%

# %% [markdown]
# Continuando con esta forma de declarar las variables, también podemos asignar

# %%

# %%

# %%

# %% [markdown]
# Observa lo que ocurre si escribes

# %%

# %% [markdown]
# ¿De qué tipo es la variable división?. Teclea type(división)

# %%

# %%

# %% [markdown]
#  Python reconoce a la variable w como un entero.

# %%

# %%

# %% [markdown]
# #### Por tu cuenta
#
# Calcula la suma de 10.8, 12.2 y 0.2, guardala en una variable llamada b, y muestra la variable

# %%

# %% [markdown]
# ### 2.2.2.- Aritmetica

# %% [markdown]
# ##### Multiplicación
#
# Realiza un 7 * 4

# %%
7 * 4

# %% [markdown]
# ##### Exponenciación
#
# Realiza un 2 ** 4

# %%
2 ** 4

# %% [markdown]
# ##### División Verdadera
#
# Divide 8/5

# %%
8 / 5

# %% [markdown]
# ##### División de piso
#
# Divide 20//6 - verás como te da un rsultado entero

# %%
20 // 6

# %% [markdown]
# ##### Restante
#
# Ahora obten el restante de dividir 26 / 8 usando el operador %

# %%
26 % 8

# %% [markdown]
# ##### División de piso
#
# Divide 20//6 - verás como te da un resultado entero

# %%
20 // 6

# %% [markdown]
# ##### Agrupación con parentesis
#
# Checa que python respeta la jerarquia de operaciones

# %%

# %% [markdown]
# #### Por tu cuenta
#
# Evalua la expresión 3*(4-5) con y sin parentesis. Son los paréntesis redundantes?

# %% [markdown]
# ---

# %% [markdown]
# ## 2.3.- Función print, arreglos con comillas simples y dobles comillas

# %% [markdown]
# ### 2.3.1.- La función Print

# %% [markdown]
# ejecuta el siguiente snippet
#
# **print(“Hoy aprenderé a usar la función print”)**

# %%

# %%

# %% [markdown]
# Ahora Ejecuta
#
# **Print(“Hoy aprenderé a usar la función print”)**

# %%

# %%

# %% [markdown]
# Por ejemplo, para declarar el arreglo
# Hoy aprenderé a usar la función ‘print’

# %% [markdown]
# **print(“Hoy aprenderé a usar la función ‘print’”)**

# %%
var = "Today I will learn python in a hyper marathon"
print(type(var))

# %% [markdown]
# ### 2.3.1.- Triples comillas
#
# Esto se resuelve utilizando dobles comillas

# %%

# %% [markdown]
# Esta diferencia será más obvia cuando trabajemos líneas de texto en el ambiente de JupyterLab.
# Otras opciones para insertar texto el prompt de Anaconda son
#
# **print(‘Hoy aprenderé’,’a usar’, ‘la función print’)**
#
# **print(‘Hoy aprenderé\n a usar\n la función print’)**
#
#
# print(‘También aprenderé a dividir \
#
# …: líneas de texto cuando las expresiones \
#
# …: sean demasiado largas’)

# %%

# %%

# %%

# %%

# %% [markdown]
# Cuando requieras utilizar comillas dobles y sencillas en algún enunciado, por ejemplo
#
# Aprender ‘Python’, es realmente “sencillo”
#
# Es posible hacerlo utilizando triples comillas

# %%

# %%

# %% [markdown]
# Con Python, puedes asignar a una variable un arreglo de caracteres
#
# **arreglo1=‘Hoy aprenderé a usar la función print’**

# %%

# %%

# %%

# %% [markdown]
# Con estas herramientas, ya puedes mostrar los resultados de los cálculos en una forma más amigable
#
# Arma un programa que divida 2 variables y devuelva "w elevado a la z da por resultado y"

# %%

# %%

# %%

# %% [markdown]
# ### 2.3.1.- Obtener input del usuario
#
# Input que solicita al usuario la información específica que debe ingresar al programa
#
# **nombre=input(‘¿Cuál es tu nombre?’)**

# %%

# %%

# %% [markdown]
# Posteriormente puedes llamar a la variable nombre

# %% [markdown]
# Por default, Python guarda las entradas que recibe del comando input como un arreglo. Checa el type en cuestión

# %%

# %%

# %%

# %% [markdown]
# Arma un programa que te pida 2 números y los sume

# %%

# %% [markdown]
# No se pudo, para que python pueda hay que usar **int()**

# %%

# %% [markdown]
# #### Por tu cuenta
#
# Usa float() para convertir "6.2" (una cadena) a un valor flotante

# %%

# %%

# %%

# %% [markdown]
# ---

# %% [markdown]
# ## 2.4.- Primeros Programas con python

# %% [markdown]
# Ejecuta el script ejemplo1_c2.py

# %% [markdown]
# Recuerda que en python también podemos hacer comparaciones mayor y menor que

# %% [markdown]
# Y que también, para cehcar si algo es igual, se usa ==

# %%
x = 3
y = 2
if (x == y):
    print("x and y are the same")
else:
    print("x and y are not the same")

# %% [markdown]
# ---

# %% [markdown]
# # Control Structures

# %%
age = int(input("Inset your age: "))

if (age < 18):
    print("You are underage")
elif (age > 60):
    print("You are old")
else:
    print("You are an adult")

# %%
for i in "Python":
    print(i)

# %%
list = [
    [0, 2, 3],
    [23, 2, 3],
]

for i in list:
    print(i)
    for j in i:
        print(j)

# %%
subjects = ["FC", "MD", "QM"]
grades   = [2, 3, 4]

for i, j in zip(subjects, grades):
    print("The grade of ", i, "is ", j)

# %%
def f(x): return (x ** 2)

for i in range(0, 5, 1):
    print([i, f(i)])

# %%
A = [
    [1, 2],
    [3, 4],
]

B = [
    [5, 6],
    [7, 8],
]

print(A[0][0])
print(B)

# %%
def MatrixSum(A, B):
    C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[i])):
            C[i][j] = A[i][j] + B[i][j]
    return C

def MatrixSub(A, B):
    C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[i])):
            C[i][j] = A[i][j] - B[i][j]
    return C

def MatrixProd(A, B):
    C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                C[i][j] += A[i][k] * B[k][j]
    return C


# %%
MatrixSum(A, B)

# %%
MatrixSub(A, B)

# %%
MatrixProd(A, B)
