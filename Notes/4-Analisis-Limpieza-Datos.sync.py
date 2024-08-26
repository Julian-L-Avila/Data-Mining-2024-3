# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="06ivyimCUpbD"
# ### Preparación de Datos
#
# La preparación de datos o limpieza de datos es un proceso que permite analizar la información para detectar y corregir los errores que determinado dataset pueda presentar (E.g. datos nulos). Estos errores se pueden clasificar en:
# - **Tipos de datos**: representación númerica de un datos que tiene un significado categórico (E.g. números que reprentan un estado civil o un color).
# - **Datos categóricos**: problemas de escritura en los datos de tipo categórico que pueden causar conflictos (E.g. mismo valor en el campo pero diferencias entre mayúsculas y minúsculas => RoJO y rojo).
# - **Uniformidad de los datos**: mismo formato para todos los valores de un campo (E.g. fechas con un único formato).
# - **Completitud de los datos**: presencia de datos nulos, información faltante.
#
# Esta limpieza de datos es importante para poder tener datasets de buena calidad de forma que los resultados de los procesos que se quieran realizar sobre los datos sean acertados.
#
# Para este ejemplo vamos a utilizar el siguiente dataset: https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings. Este dataset, llamado **Video Game Sales with Ratings**, contiene información de ventas de un listado de videojuegos incluyendo los nombres de los videojuegos, la plataforma, el año de publicación, el género, el editor, el desarrollador, sus rankings entre otros. Si se desea hacer un análisis de datos sobre este dataset es importante primero realizar una limpieza de datos.
#
# **Nota de interés**: este dataset que veremos en este ejemplo es un dataset construído utilizando Web Scraping, una técnica de obtención de datos, a partir de un sitio web. Este tema lo veremos en próximas unidades.
#
# #### Pasos importantes para realizar la limpieza de los datos:
# 1. Conocer toda la información general del dataset. Para este ejemplo, como ya se mencionó, se utilizará el dataset **Video Game Sales with Ratings** que contiene los datos de ventas de un listado de más de 10K videojuegos, todos ellos con más de 100K copias vendidas. Las columnas que contiene este dataset son las siguientes:
#    - **Name**: nombre del videojuego. Tipo categórico.
#    - **Platorm**: nombre de la plataforma destino del videojuego. Tipo categórico.
#    - **Year_of_Release**: año de publicación del videojuego. Tipo numérico.
#    - **Genre**: género del videojuego. Tipo categórico.
#    - **Publisher**: editor del videojuego. Tipo categórico.
#    - **NA_Sales**: número de ventas en Norte América. Tipo numérico.
#    - **EU_Sales**: número de ventas en Europa. Tipo numérico.
#    - **JP_Sales**: número de ventas en Japón. Tipo numérico.
#    - **Other_Sales**: número de ventas en otras regiones: Tipo numérico.
#    - **Global_Sales**: número de ventas a nivel global: Tipo numérico.
#    - **Critic_Score**: puntaje compilado de las críticas al videoojuego en Metacritic. Tipo numérico.
#    - **Critic_Count**: número de críticas realizadas al videojuego en Critic_Score. Tipo numérico.
#    - **User_Score**: puntaje dado por los usuarios de Metacritic al videojuego. Tipo numérico.
#    - **User_Count**: numéro de usuarios que dieron su puntaje al videojuego en User_Score. Tipo numérico.
#    - **Devloper**: desarrollador del videojuego. Tipo categórico.
#    - **Rating**: Rating ESRB. Tipo categórico.
#
# 2. Realizar un análisis exploratorio de los datos para identificar los problemas o errores presentes en el dataset y pensar en las posibles estrategias para solucionar dichos inconvenientes.
# 3. Aplicar las estrategias de limpieza de datos.
#

# %% [markdown] id="OMSZh0AwUpbY"
# ### Una vez conocida la información general del dataset ¡podemos comenzar con el análisis exploratorio!
#
# Para realizar el análisis exploratorio y la posterior limpieza de los datos vamos a utilizar el lenguaje de programación Python, específicamente su módulo **pandas**. Para comenzar debemos importar dicho módulo.

# %% id="Dmbyhv-oUpbZ" executionInfo={"status": "ok", "timestamp": 1708963056467, "user_tz": 300, "elapsed": 368, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Importar el o los módulos necesarios para analizar y limpiar los datos.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown] id="XFHrSd-DUpbc"
# A continuación debemos importar el dataset con el que vamos a trabajar, en este caso el dataset **Video Game Sales with Ratings**. Para importarlo debemos descargar el archivo *.csv*, ubicado en del siguiente link https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings. A continuación debemos ubicarlo en el mismo directorio donde se encuentra ubicado este *notebook*. Por último usaremos pandas para importar el dataset y asignarlo a una variable.

# %% id="P6N_muxBUpbd" executionInfo={"status": "ok", "timestamp": 1708963194632, "user_tz": 300, "elapsed": 169, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Para importar el dataset y guardarlo en una variable usamos el método read_csv().
original_df = pd.read_csv('./Data/Video_Games_Sales_as_at_22_Dec_2016.csv')

# %% [markdown] id="46Ubee7rUpbd"
# Si en el dataset que importamos no están especificados los nombres de las columnas podemos usar el parámetro **header=None** en el método de pandas *read_csv()*, esto con el fin de evitar que el primer registro de dataset sea tratado como las columnas del mismo (E.g. esto puede pasar para archivos de extensión .data). Con el dataset guardado en la variable *original_df* podemos visualizar los datos.

# %% id="DJAenhnuUpbe" outputId="07c18dc5-fc3b-4c84-bbc3-2727a5939895" colab={"base_uri": "https://localhost:8080/", "height": 365} executionInfo={"status": "ok", "timestamp": 1708964692235, "user_tz": 300, "elapsed": 183, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Paa visualizar los datos usamos el método head(). Esta función, por defecto, nos mostrará los datos de los primeros
# cinco registros del dataset. Si queremos ver más registros debemos indicarle al método el número de registros que
# queremos visualizar. E.g. head(8).
original_df.tail(20)

# %% [markdown] id="BCNOCTXxUpbf"
# Con esta primera visualización podemos notar que las columnas están correctamente nombradas. En el dado caso en que las columnas no tengan los nombres apropiados, estos nombres se pueden asignar usando el atributo *columns*. E.g. *df.columns = [...]*, donde en el arreglo se especifican los nombres por medio de strings (texto) usando comillas dobles ("") o comillas simples ('').
#
# Ahora, si queremos en algún momento visualizar los últimos elementos del dataset podemos usar el método *tail()*. También podemos visualizar un grupo aleatorio de registros usando el método *sample()* al que le podemos pasar como parámetro un valor *n* que define el tamaño de dicha grupo o muestra aleatoria.
#
# Lo siguiente que podemos hacer es visualizar la información de cada una de las columnas.

# %% id="AOqjPRpDUpbf" outputId="b5789bb6-387c-42f0-d7e5-177dee61b3df" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708963402953, "user_tz": 300, "elapsed": 170, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el atributo shape para observar el tamaño del dataset.
original_df.shape

# %% [markdown] id="ZwEXh7JKUpbg"
# Esto nos está mostrando que el dataset cuenta con 16719 registros (filas) y 16 columnas. Usando el método *info()* podemos obtener la información de las columnas.

# %% id="dbgFduKaUpbh" outputId="8cbf509b-56ab-4408-f063-36ec3dc20416" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708963435596, "user_tz": 300, "elapsed": 188, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método info() para obtener información de las columnas del dataset.
original_df.info()

# %% [markdown] id="AyEkWdwpUpbh"
# Al verificar esta información podemos comenzar a detectar los primeros problemas. Anteriormente habíamos dicho que el número total de registros era de 16719, pero aquí vemos que para algunas columnas los registros no están completos. Eso quiere decir que en cada una de esas columnas hay valores nulos. También es raro que la columna *User_Score* sea de tipo *object*, pues debería tener un tipo numérico por representar un puntaje.
#
# Otra forma de obtener información valiosa de los datos de cada columna es utilizar el método *describe()*. Este método nos provee información estadística de los datos de cada columna como la media, la desviación estándar, cuartiles, etc, mismos que nos pueden mostrar el comportamiento de sus respectivos datos. Se debe tener en cuenta que este método actúa por defecto sobre las columnas de tipo númerico.

# %% id="KBpAwk7IUpbi" outputId="5e5a5d76-7423-48e9-b7dd-e327d3dce9c7" colab={"base_uri": "https://localhost:8080/", "height": 332} executionInfo={"status": "ok", "timestamp": 1708963563198, "user_tz": 300, "elapsed": 192, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método descibre para obtener datos estadísticos de las columnas de tipo numérico.
original_df.describe().T

# %% [markdown] id="nwBKSlnPUpbi"
# El método *describe()* nos está mostrando información interesante. Por ejemplo en la columna *User_Count* podemos ver los valores 10.00, 24.00 y 81.00 para los cuartiles 25%, 50% y 75% respectivamente. Pero también nos está mostrando un valor de 10665.00 para el valor máximo, lo que parece indicar un valor atípico.
#
# Para verificar este comportamiento podemos utilizar una gráfica de cajas y bigotes sobre la columna *User_Count*.

# %% id="nCecDFpcUpbi" outputId="a9200f56-0ab8-4bb9-f465-a43b6451c2ce" colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"status": "ok", "timestamp": 1708963856699, "user_tz": 300, "elapsed": 481, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Obtenemos la columna, la guardamos en una variable y con el método dropna() quitamos los valores nulos
# para que al dibujar la gráfica de cajas y bigotes no nos dé ningún problema.
user_count = original_df['User_Count'].dropna()

print(type(user_count))
print(user_count.dtype)

# Dibujamos la gráfica de cajas y bigotes.
plt.boxplot([user_count])
plt.show()

# %% [markdown] id="Lt6axQIxUpbj"
# La gráfica nos muestra que efectivamente hay valores atípicos bastante importantes denominados outliers. Estos outliers pueden llegar a afectar mucho la estadística por lo que en algún punto toca reemplazarlos.
#
# Por otro lado, también es posible ver información de las columnas de tipo categórico utilizando el método *describe()*.

# %% id="0EuQWOC2Upbj" outputId="c4a971bd-d0b8-4871-ddd0-b64e76d10976" colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"status": "ok", "timestamp": 1708963910477, "user_tz": 300, "elapsed": 190, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Si usamos el método describe() con el parámetros include='0' podemos ver información adicional de las columnas
# que tienen valores categóricos.
original_df.describe(include='O').T

# %% [markdown] id="MrBdSn8SUpbj"
# Aquí obtenemos información como el conteo total de registros para cada columna (no se cuentan los valores nulos), el conteo de valores únicos para cada columna, el valor que más aparece para cada columna, etc. Viendo la tabla podemos identificar otro problema: la columna *User_Score* posee valores no númericos como *tbd*. Esto es lo que está convirtiendo la columna en tipo *object*.
#
# Algo interesante a destacar es que si se hace una investigación sobre el **Rating ESRB**, este sólo contiene seis categorías (E, E10+, T, M, AO y RP) y aquí se nos está mostrando que hay ocho valores únicos para la columna *Rating*. Verifiquemos cuáles son dichos valores únicos.

# %% id="Pl7OY9u8Upbk" outputId="3e68ffd6-b0a1-4cd4-f2e3-1fa178d92617" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708964165914, "user_tz": 300, "elapsed": 207, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método unique() sobre la columna deseada para verificar cúales son los valores únicos.
original_df['Rating'].unique()

# %% [markdown] id="DCNLhvZkUpbk"
# En este punto podemos observar que además de las categorías antes mencionadas hay una categoría adicional denominada *K-A*. Resulta que esta categoría fue reemplazada en 1998 por la categoría *E*, es decir *K-A* ya no está vigente.

# %% [markdown] id="lPG3yKN7Upbk"
# #### Resultados del análisis exploratorio
#
# Los siguientes son los resultados que pudimos obtener a partir del análisis exploratorio que realizamos sobre el dataset:
# * Presencia de valores nulos en varias columnas.
# * La columna *User_Score* contiene elementos string que la transforman en un tipo object (categórico). Esta columna debería ser de tipo numérico.
# * Presencia de valores atípicos (E.g. columna *User_Count*).
# * La columna *Rating* contiene un valor que ya no se encuentra vigente en el Rating ESRB.
#

# %% [markdown] id="wGQh0M3FUpbl"
# ### Con los anteriores resultados ya podemos comenzar a hacer la limpieza de datos
#
# #### Valores nulos
# Comencemos tratando de solucionar el tema de los valores nulos. Muchas veces al querer solucionar este problema, la primera acción que nos sentimos tentados de realizar es borrar todos aquellos registros que tengan valores nulos y no siempre esa es la mejor solución. Primero se debería tratar de hacer un análisis de esos valores. Echemos un vistazo al número de valores nulos que tenemos por columna.

# %% id="NLQw9O2LUpbl" outputId="e8a8bad9-0df2-4533-e0e4-7db7ca67d581" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708964378059, "user_tz": 300, "elapsed": 184, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Verificamos el total de valores nulos en cada columna.
original_df.isna().sum()

# %% [markdown] id="QjpKvBSDUpbl"
# Viendo estos datos hay varias cosas a tener en cuenta:
# 1. Existen dos registros cuyo nombre del videojuego no existen. No tiene sentido conservarlos pues no sabemos de que videjuegos se tratan. Esos dos registros se pueden borrar.
# 2. El año de publicación y el nombre de los editores son dos temas complejos que podríamos dejar para el final de esta parte de los nulos.
# 3. Para las columnas *Critic_Score*, *Critic_Count* podemos usar estrategias para reemplazar los valores nulos, por ejemplo reemplazar por la media, por la moda o por la mediana.
# 4. Para la columna *User_Score* debemos atender primero el problema del error de tipo.
# 5. Para la columna *User_Count* podemos aplicar las mismas estrategias del punto 3, pero antes debemos ver el problema de los datos atípicos.
# 6. Para el caso del nombre del desarrollador, por ser una columna cuyos valores son categóricos, se podría pensar en reemplazar por la moda, pero esto no sería tan apropiado como pensamos.
# 7. Para el caso del rating podríamos reemplazar por la moda, pero antes debemos corregir el tema del valor que ya no está vigente y reemplazarlo.
#
# Comencemos pues eliminando los dos registros que no tienen un valor para el nombre del videojuego.

# %% id="L737qO1_Upbl" outputId="5aaa4190-a780-442b-f6a2-5bdf93669f5e" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708964750472, "user_tz": 300, "elapsed": 199, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Creamos una copia del dataframe original usando el método copy(). Esto lo hacemos porque más adelante podríamos necesitar
# los datos originales para realizar alguna tarea.
df_toclean = original_df.copy() # Limpiaremos los datos usando el dataframe df_toclean.

# Utilizamos el método notnull() que nos permite hacer un filtro de todos aquellos registros que no tengan un valor nulo
# para una columna en específico. Reasignamos el valor de df con el resultado del filtro.
df_toclean = df_toclean[df_toclean['Name'].notnull()]

df_toclean = df_toclean.reset_index() # Es sumamente importante resetear el índice de los datos.
df_toclean.drop('index', axis=1, inplace=True)


# Creamos una copia a partir de df_toclean. Esta copia la usaremos a modo de backup para guardar algunos avances (distintos
# a la imputación de valores nulos en columnas númericas) que nos conviene tener para más adelante.
df_aux = df_toclean.copy()

# Verificamos nuevamente el número de nulos.
df_toclean.isna().sum()

# %% [markdown] id="IkWzSzG1Upbm"
# Confirmamos que ya no tenemos registros con valores nulos para el nombre del videojuego. Además nos damos cuenta que dichos registros también eran los que tenían valores nulos en el género pues dichos nulos han desaparecido.
#
# Pasemos al caso de las columnas *Critic_Score* y *Critic_Count*. Estas columnas no tienen valores atípicos importantes por lo que podemos proceder realizando alguna estrategia de imputación (reemplazar por media, mediana o moda). Veamos algunas reglas y sugerencias:
# * La **media** o promedio es un punto de equilibrio o centro de masas del conjunto de datos. Su cálculo se hace haciendo la sumatoria de los datos y diviendo el resultado entre el número total de datos del conjunto. Esta métrica se comporta muy bien cuando los datos son homogéneos, es decir, tienen una distribución normal.
# * La **mediana** es un valor que deja por debajo de sí a la mitad de los datos y por encima de sí a la otra mitad (los datos deben estar ordenados). Funciona mucho mejor que la media si los datos son heterogéneos, es decir no tienen una distribución normal.
# * La **moda** es el dato que más se repite en un conjunto de datos. Se recomienda usar en conjuntos de datos de tipo categórico.
# * En un conjunto de datos con distribución normal se puede utilizar cualquiera de las tres métricas para hacer imputación.
#
# Veamos entonces cómo están distribuidos los datos en estas dos columnas.

# %% id="2RwcwOe-Upbm" outputId="a9e6cb59-a5bf-41f3-a6b9-dbca8ef31cba" colab={"base_uri": "https://localhost:8080/", "height": 433} executionInfo={"status": "ok", "timestamp": 1708964951622, "user_tz": 300, "elapsed": 505, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Obtenemos la columna, la guardamos en una variable y con el método dropna() quitamos los valores nulos
# para que al dibujar la gráfica de histograma no nos dé ningún problema.
critic_score = df_toclean['Critic_Score'].dropna()

# Usamos la gráfica de histograma para ver la distribución de los datos.
plt.hist(critic_score)
plt.show()

# %% id="xhyRgVukUpbn" outputId="464f0c59-d3d3-470e-a79b-74e6c68edf61" colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"status": "ok", "timestamp": 1708964982973, "user_tz": 300, "elapsed": 523, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Obtenemos la columna, la guardamos en una variable y con el método dropna() quitamos los valores nulos
# para que al dibujar la gráfica de histograma no nos dé ningún problema.
critic_count = df_toclean['Critic_Count'].dropna()

# Usamos la gráfica de histograma para ver la distribución de los datos.
plt.hist(critic_count)
plt.show()

# %% [markdown] id="UGLFqlq7Upbn"
# Estas gráficas nos indican que para las columnas *Critic_Score* y *Critic_Count* la distribución de sus datos no parece ser homogénea por lo que es conveniente utilizar la mediana como estrategia de imputación de sus valores nulos.

# %% id="0ziADHGvUpbn" outputId="3747e018-ba2f-4832-f6b8-c96d96bf192e" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708965107605, "user_tz": 300, "elapsed": 5827, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método median() para obtener la mediana de los valores de cada columna.
cs_median = critic_score.median()
cc_median = critic_count.median()

# Con el método fillna() reemplazamos los valores nulos de cada columna por la mediana.
df_toclean.fillna({ 'Critic_Score': cs_median }, inplace=True)
df_toclean.fillna({ 'Critic_Count': cc_median }, inplace=True)

# Verificamos el número total de valores nulos de cada columna del dataset.
df_toclean.isna().sum()

# %% [markdown] id="6eH-HAkmUpbo"
# Por el momento esto es lo único que podemos hacer con respecto a los valores nulos pues para resolver el caso de las demás columnas debemos atender antes otros problemas.
#
# #### Error de tipos en la columna *User_Score*
# Atendamos entonces el caso de la columna *User_Score* que contiene elementos string cuando esta columna debería ser de tipo númerico. Verifiquemos cuáles son los valores únicos que contiene esta columna.

# %% id="QK9G_qquUpbo" outputId="a8fbf45a-0a24-4848-edf1-615d4663837b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708965250416, "user_tz": 300, "elapsed": 203, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método unique() para ver los valores únicos en la columna.
df_toclean['User_Score'].unique()

# %% [markdown] id="52aqiZgoUpbo"
# Como se puede ver tenemos un valor que no representa ningún dato númerico, *tbd*. Para solventar esta situación vamos a transformar los valores de esta columna a tipo float usando el método *to_numeric()*, indicando que aquellos valores string que no se puedan parsear a un tipo númerico tomen el valor nulo.

# %% id="DPfYqhDbUpbo" outputId="99ba7c41-28d5-4c66-951b-f1951c0c3e02" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708965381975, "user_tz": 300, "elapsed": 187, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método de pandas to_numeric() para transformar los datos de una columna a valores numéricos
# El parámetro errors='coerce' indica que los valores que no puedan ser parseado tomen el valor nulo.
df_toclean['User_Score'] = pd.to_numeric(df_toclean['User_Score'], errors='coerce')

# Usamos de nuevo info para verificar el cambio.
df_toclean.info()

# %% [markdown] id="EoVZH_kPUpbp"
# Efectivamente la columna *User_Score* ahora es de tipo float y como se puede ver el número de valores nulos ha aumentado con respecto a la última verificación hecha con el método *info()*, lo que indica que la transformación se ha hecho de manera correcta. Algo que podemos hacer en este punto es verificar, mediante un gráfico de cajas y bigotes, si para la columna *User_Score* existen datos atípicos.

# %% id="t1cH3ZbYUpbp" outputId="9f468c61-6fdd-4646-c9da-09a60e5872dc" colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"status": "ok", "timestamp": 1708966022348, "user_tz": 300, "elapsed": 198, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Guardamos el avance de haber transformado el tipo de dato de la columna de manera correcta.
df_aux['User_Score'] = df_toclean['User_Score']

# Obtenemos la columna, la guardamos en una variable y con el método dropna() quitamos los valores nulos
# para que al dibujar la gráfica de cajas y bigotes no nos dé ningún problema.
user_score = df_toclean['User_Score'].dropna()

# Dibujamos la gráfica de cajas y bigotes.
plt.boxplot([user_score])
plt.show()

# %% [markdown] id="7RtEsbpEUpbp"
# En la gráfica podemos identificar algunos datos atípicos que no son muy relevantes pues no son outliers ya que se encuentran en cercanías al valor mínimo representado por el bigote inferior. Esto quiere decir que esta columna esta ya lista para reemplazar su valores nulos usando alguna estrategia de imputación.

# %% [markdown] id="53LvTf-FUpbq"
# #### Valores atípicos en la columna *User_Count*
#
# Sabemos que en la columna *User_Count* hay una gran cantidad de outliers que debemos limpiar para no afectar en gran medida las estadísticas del dataset para futuros procesos sobre los datos. La estrategia que podemos seguir es calcular los outliers y reemplazarlos por valores nulos que luego limpiaremos.
#
# Para calcular los outliers primero debemos calcular los cuartiles, el rango intercuartil y los valores de los bigotes (estos bigotes se refieren a la gráfica de cajas y bigotes) correspondientes a los datos de esta columna.

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="zCsgfL9GUB8I" executionInfo={"status": "ok", "timestamp": 1708966062786, "user_tz": 300, "elapsed": 416, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}} outputId="d375be64-3ad7-4d56-dd65-67ac01d0ecde"
# Guardamos el avance de haber transformado el tipo de dato de la columna de manera correcta.
df_aux['User_Count'] = df_toclean['User_Count']

# Obtenemos la columna, la guardamos en una variable y con el método dropna() quitamos los valores nulos
# para que al dibujar la gráfica de cajas y bigotes no nos dé ningún problema.
user_count = df_toclean['User_Count'].dropna()

# Dibujamos la gráfica de cajas y bigotes.
plt.boxplot([user_count])
plt.show()

# %% id="bBwwBkjuUpbq" outputId="d80880ec-4bb0-49d5-c427-894b4496696c" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708966172940, "user_tz": 300, "elapsed": 184, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método quantile() para obtener los cuartiles 1 (25%) y 3 (75%)
uc_q1 = df_toclean['User_Count'].quantile(0.25)
uc_q3 = df_toclean['User_Count'].quantile(0.75)

print(uc_q1)
print(uc_q3)

# Calculamos el rango intercuartil
uc_iqr = uc_q3 - uc_q1

# Calculamos el valor de los bigotes inferior y superior
uc_lw = uc_q1 - (1.5 * uc_iqr)
uc_uw = uc_q3 + (1.5 * uc_iqr)

# Verificamos los resultados
print('Primer cuartil: ', uc_q1)
print('Tercer cuartil: ', uc_q3)
print('Rango intercuartil: ', uc_iqr)
print('Bigote inferior: ', uc_lw)
print('Bigote superior: ', uc_uw)

# %% [markdown] id="CFyoMv-bUpbq"
# Teniendo los valores de los bigotes, podemos detectar los outliers. Estos outliers se pueden definir como todos aquellos valores que se encuentran por fuera del rango comprendido entre los bigotes inferior y superior. Todos estos valores los vamos a volver nulos para tratar de normalizar los datos.

# %% id="78P7zQZGUpbq" outputId="c8a2c8fd-5769-4fbf-805d-9b7b58de759f" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708966402778, "user_tz": 300, "elapsed": 233, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Armamos una condición lógica y la guardamos en una variable.
uc_outliers = (df_toclean['User_Count'] < uc_lw) | (df_toclean['User_Count'] > uc_uw)

# Usamos el atributo loc para acceder a todos los registros que cumplan la condición especificada y reemplazamos todos los
# valores de dichos registros para la columna elegida con el valor nulo.
df_toclean.loc[uc_outliers, 'User_Count'] = np.nan

# Guardamos el avance del manejo de outliers para la columna.
df_aux['User_Count'] = df_toclean['User_Count']

# Volvemos a usar los métodos isna() y sum() para verificar los valores nulos de cada columna.
df_toclean.isna().sum()

# %% [markdown] id="_a22pOSsUpbq"
# Como podemos observar el número de valores nulos de la columna *User_Count* aumentó en casi mil unidades, por lo que la tranformación que hicimos surtió efecto. Con esto tenemos la columna lista para reemplazar todos sus valores nulos.

# %% [markdown] id="zLPvW_mnUpbr"
# #### Reemplazar el valor no vigente de la columna *Rating*
#
# Para el caso de la columna *Rating* debemos cambiar el valor no vigente *K-A* por su reemplazo *E*.

# %% id="Logz6PlvUpbr" outputId="feb63bdb-c0a5-41a0-9a50-451bc1be3ad5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708966517073, "user_tz": 300, "elapsed": 189, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método replace() de pandas para cambiar los valores de una columna.
df_toclean = df_toclean.replace({ 'Rating': { 'K-A': 'E' }})

# Guardar el avance del cambio del valor no vigente de la columna.
df_aux['Rating'] = df_toclean['Rating']

# Usamos de nuevo el método unique() para verificar el cambio.
df_toclean['Rating'].unique()

# %% [markdown] id="FaYsST3zUpbr"
# ¡En este punto ya tenemos todas las columnas listas para solucionar el problema de los valores nulos!

# %% [markdown] id="8L5n2m-2Upbr"
# #### Volviendo a los nulos
#
# Retomando el problema de los valores nulos, recordemos las columnas que nos hacen falta por limpiar:
# 1. La columna *User_Score*.
# 2. La columna *User_Count*.
# 3. La columna *Rating*.
# 4. La columna *Developer*.
# 5. La columna *Year_of_Release*.
# 6. La columna *Publisher*.
#
# Comencemos por la columna *User_Score*. Para saber que estrategía de imputación vamos a utilizar veamos la distribución de sus datos.

# %% id="BwUTc6-QUpbs" outputId="ab5634cd-34c8-4d81-fcab-f21cdc521d64" colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"status": "ok", "timestamp": 1708966651013, "user_tz": 300, "elapsed": 765, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos la variable user_score para dibujar la gráfica de histograma.
plt.hist(user_score)
plt.show()

# %% [markdown] id="66ICgGexUpbs"
# Este histograma nos muestra una forma de campana desviada hacia la derecha, por lo que la mejor estrategia de imputación para esta columna es reemplazar los valores nulos por la mediana.

# %% id="TQNP3WraUpbs" outputId="3bf05366-7816-421e-e967-87ac6f3d4cc7" colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"status": "ok", "timestamp": 1708966654405, "user_tz": 300, "elapsed": 619, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método median() para obtener el valor de la mediana y lo guardamos en una variable.
us_median = user_score.median()

# Con el método fillna() reemplazamos los valores nulos de una columna por un nuevo valor.
df_toclean.fillna(value={'User_Score': us_median }, inplace=True)

# Verificamos nuevamente el número de nulos.
df_toclean.isna().sum()

plt.hist(user_score)
plt.show()

# %% [markdown] id="MW6PxCg6Upbt"
# Prosigamos con la columna *User_Count*. Verifiquemos también la distribución de sus datos.

# %% id="wTpsg64yUpbt" outputId="4dc63bb7-a733-411f-bebd-5f80fb62ce10" colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"status": "ok", "timestamp": 1708966679719, "user_tz": 300, "elapsed": 636, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Obtenemos la columna, la guardamos en una variable y con el método dropna() quitamos los valores nulos
# para que al dibujar la gráfica de histograma no nos dé ningún problema.
user_count = df_toclean['User_Count'].dropna()

# Usamos la gráfica de histograma para ver la distribución de los datos.
plt.hist(user_count)
plt.show()

# %% [markdown] id="oz4G1LX5Upbx"
# La gráfica nos está indicando que la distribución de los datos para la columna *User_Count* no es homogénea por lo que aquí también es conveniente imputar los datos usando la mediana de los valores de esta columna.

# %% id="MiLWhERzUpbz" outputId="dcf8c63a-b108-46e0-c057-7b935ac94ed1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708966683459, "user_tz": 300, "elapsed": 200, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método median() para obtener el valor de la mediana y lo guardamos en una variable.
uc_median = user_count.median()

# Con el método fillna() reemplazamos los valores nulos de una columna por un nuevo valor.
df_toclean.fillna({ 'User_Count': uc_median }, inplace=True)

# Verificamos nuevamente el número de nulos.
df_toclean.isna().sum()

# %% [markdown] id="82uszCKMUpb0"
# Para la columna *Rating* lo que podemos hacer es utilizar la moda para reemplazar los valores nulos. En este caso sabemos que el valor que más se repite es el rating *E*, por lo que reemplazamos los nulos por este valor.

# %% id="2WP3WBOCUpb0" outputId="b2b73b9c-e3dc-44a8-f834-f0a57c73d7da" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1708966726542, "user_tz": 300, "elapsed": 188, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos método fillna() para reemplazar los valores nulos de una columna por un nuevo valor.
df_toclean.fillna(value={ 'Rating': 'E' }, inplace=True)

# Guardamos el avance de la imputación de los valores nulos de la columna categórica.
df_aux['Rating'] = df_toclean['Rating']

# Verificamos nuevamente el número de nulos.
df_toclean.isna().sum()

# %% [markdown] id="uUi5M-YsUpb1"
# En este punto ya solo nos quedan tres columnas con valores nulos. Veamos primero el caso de la columna *Developer*. Como habíamos identificado anteriormente, esta columna es de tipo categórico por lo que, siguiendo las recomendaciones de imputación, podríamos reemplazar sus valores nulos con la moda. Pero aquí surge un inconveniente: si reemplazamos con la moda nos pueden quedar desarrolladores equivocados para muchos de los videojuegos de la lista (E.g. Nintendo como desarrollador Call of Duty). Aquí lo que podemos hacer es reemplazar los valores nulos por el valor correspondiente al editor para cada registro. Esto porque es bastante común que la empresa que publica el videojuego también sea la misma que lo desarrolla (E.g. aveces se pueden ver a Activision como el desarrollador de Call of Duty).

# %% id="dmHEK2mDUpb2" outputId="8d6ba965-a89b-4e64-8380-ee4e6ef64662" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1679939748549, "user_tz": 300, "elapsed": 844, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Sabemos que el total de registros del dataset es de 16717 por lo que vamos a recorrer esa cantidad.
for i in range(16717):
    developer = df_toclean['Developer'][i]
    publisher = df_toclean['Publisher'][i]

    # Aquí vamos a verificar que el valor de Developer para esa posición sea nulo y que el valor de Publisher para esa misma
    # posición sea diferente de nulo. Si estas dos condiciones se cumplen hacemos el reemplazo.
    if pd.isna(developer) and not pd.isna(publisher):
        df_toclean.loc[i, 'Developer'] = publisher

# Guardamos el avance de la imputación de los valores nulos de la columna categórica.
df_aux['Developer'] = df_toclean['Developer']

df_toclean.isna().sum()

# %% [markdown] id="Wa1amdWiUpb2"
# Con esta estrategia hemos podido eliminar una gran cantidad de valores nulos de la columna *Developer*. Ya en este punto no tenemos una estrategia clara para eliminar los nulos faltantes de dicha columna. Veamos las dos columnas restantes:
# - Si analizamos la columna *Publisher* aquí tampoco podemos aplicar la moda por la misma razón que no pudimos hacerlo para la columna *Developer*. Por otro lado, la estrategia usada para para eliminar los valores nulos de *Developer* no nos serviría de mucho pues no es muy común que la empresa desarrolladora sea catalogada también como la editora (E.g. Treyarch).
# - La columna *Year_of_Release*, a pesar de ser una columna numérica, en realidad sus valores son categóricos, por lo que habría que usar la moda para reemplazar los valores nulos. De nuevo esta estrategia no sería muy buena pues muchos videojuegos quedarían con un año equivocado de publicación.
#
# En ese sentido podemos optar por dos caminos:
# 1. Eliminar los registros que tienen estos valores nulos.
# 2. Reemplazar esos valores nulos por otro valor que indique que es desconocido (esto se puede ver en varias columnas categóricas del dataset como *Publisher* o *Developer*).
#
# Veamos que porcentaje del dataset representan estos últimos registros con valores nulos.

# %% id="stQsLp1FUpb3" outputId="1221b65e-f4c9-47a6-8d0c-3e99f99e9628" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1679939794061, "user_tz": 300, "elapsed": 201, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Calculamos el total de celdas que posee el dataset (filas x columnas).
total_cells = np.product(df_toclean.shape)

# Calculamos el total de celdas que tienen valores nulos.
missing = df_toclean.isna().sum() # Celdas con valores nulos por columna.
total_missing = missing.sum()

# Finalmente calculamos el porcentaje.
missing_percent = (total_missing / total_cells) * 100

print('Porcentaje de valores nulos en todo el dataset: ', missing_percent)

# %% [markdown] id="X1ggeIVqUpb3"
# Tenemos que el porcentaje de valores nulos restantes es de 14% en comparación con todos los datos del dataset. Aunque este porcentaje es aceptable para tomar la decisión de borrar los datos, debemos tener en cuenta que tenemos muchas más columnas con valores para estos registros que nos pueden brindar información valiosa.
#
# Para el caso de este ejercicio vamos a optar por el segundo camino. Lo que haremos será reemplazar los valores nulos de la columna *Year_of_Release* por un valor de 9999.00 y los valores nulos de las columnas *Publisher* y *Developer* por el valor 'Unknown' (valor que se maneja en estas columnas).

# %% id="NzDcpo78Upb4" outputId="48d41784-3dd1-421a-e123-391a0626fb38" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1679939868255, "user_tz": 300, "elapsed": 338, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método fillna() para reemplazar los valores nulos de las colmnas restantes.
df_toclean.fillna({ 'Year_of_Release': 9999.00 }, inplace=True)
df_toclean.fillna({ 'Publisher': 'Unknown' }, inplace=True)
df_toclean.fillna({ 'Developer': 'Unknown' }, inplace=True)

# Guardamos los avances referentes a las imputaciones finales de valores nulos en las columnas categóricas.
df_aux['Year_of_Release'] = df_toclean['Year_of_Release']
df_aux['Publisher'] = df_toclean['Developer']
df_aux['Developer'] = df_toclean['Developer']

# Verificamos el número total de valores nulos.
df_toclean.isna().sum()

# %% [markdown] id="dId3VVnVUpb4"
# Como podemos ver, hemos eliminado por completo los valores nulos y hemos logrado solucionar los errores que habíamos detectado. Veamos ahora información estadística de nuestras columnas numéricas ya procesadas.

# %% id="jc3LX05sUpb4" outputId="6dbc4066-9d91-4db7-d723-a2d90f1a4838" colab={"base_uri": "https://localhost:8080/", "height": 407} executionInfo={"status": "ok", "timestamp": 1679939877679, "user_tz": 300, "elapsed": 246, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Usamos el método describe() para ver información estadística de las columnas del dataset final.
df_toclean.describe().T

# %% [markdown] id="WC3XFdWpUpb4"
# #### Ups! parece que algo no anda del todo bien...
#
# Las columnas *Critic_Score*, *Critic_Count*, *User_Score* y *User_Count* parecieran tener un comportamiento extraño pues sus tres cuartiles tienen el mismo valor. Además *Critic_Count* y *User_Count* parecieran tener aún valores atípicos. Veamos por ejemplo estas columnas en una gráfica de cajas y bigotes.

# %% id="-9HjY73lUpb5" outputId="6b325a92-4ec5-4fc0-fa35-1f62295c0f88" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1679939901134, "user_tz": 300, "elapsed": 249, "user": {"displayName": "Carlos Andres Gomez Vasco", "userId": "02293680002023987679"}}
# Visualizamos las columnas usando una gráfica de cajas y bigotes.
plt.boxplot([df_toclean['Critic_Score'], df_toclean['Critic_Count'], df_toclean['User_Score'], df_toclean['User_Count']])
plt.show()

# %% [markdown] id="3XYufZheUpb5"
# Con esta gráfica confirmamos que estas columnas tienen un comportamiento extraño: todos los cuartiles tienen el mismo valor de la mediana. Recordando lo que hicimos al reemplazar los valores nulos esto tiene sentido pues hicimos una imputación usando la mediana en mas de 8K valores, esto es, más de la mitad de los registros totales para cada una de las columnas. Este comportamiento está causando que se presenten muchos valores atípicos.
#
# Lo que acabamos de ver quiere decir que pasamos algo por alto y estas cuatro columnas son suceptibles a mejoras.

# %% [markdown] id="s3itJNH2Upb6"
# ### ¿Cómo podemos mejorar el análisis y limpieza realizados?
#
# Para mejorar el análisis y limpieza que ya hicimos sobre las cuatro columnas podemos tener en cuenta los siguiente:
# 1. Para hacer la imputación usando la mediana lo que hicimos fue ver si todos los datos de la columna tenían una distribución normal y después calculamos la mediana de toda la columna. Esto es lo que se conoce como un análisis univariado (lo que no está mal). Para mejorar esto se pueden hacer dos cosas:
#    - Lo primero es utilizar un método estadístico para asegurarnos del tipo de distribución que tienen los datos de la columna. Esto porque muchas veces no se puede asegurar si la distribución es homogénea o heterogénea usando únicamente una gráfica. Se puede usar **Kolmogorov-Smirnov** para una muestra mayor a 50.
#    - La segunda alternativa es pasar a un análisis bivariado. Esto significa, tratar de encontrar una relación entre la columna a la que se quiere hacer imputación y otra columna del dataset. Con esta relación verificar su distribución y decidir que estrategia usar. E.g. Ver la distribución de User_Count para un género específico del videojuego e imputar solo los registros que pertenezcan a dicho género.
# 2. Usar un algoritmo de interpolación para imputar los valores nulos.
# 3. Crear un módelo para calcular qué valor deberían tener los valores nulos e imputarlos. Esto requiere ayuda de algoritmos de Machine Learning.
# 4. En referencia a los valores atípicos se podría ser un poco más riguroso, ejecutando el código de eliminación de dichos valores las veces que sea necesario hasta que no haya ningún valor atípico. **Ojo**: esto puede causar pérdida de información.
#
# En este ejemplo vamos a optar por la opción 1.b: trataremos de hacer análisis bivariado. ¿Recuerdan el dataframe *df_aux*? Es momento de usarlo. Este dataframe tiene guardados algunos cambios que nos pueden servir para no repetir el proceso completo. Recordemos dichos cambios:
# - Todas las columnas de tipo categórico ya están correctamente imputadas, es decir, sus valores nulos ya se reemplazaron de forma correcta.
# - La columna *User_Count* ya no tiene los outliers que habíamos detectado al principio.

# %% id="6-1V-XEIUpb6" outputId="adc17db5-17e8-4b60-8af3-586ebbb9262c"
# Verificamos en que estado está el dataframe df_aux en cuanto a sus valores nulos.
df_aux.isna().sum()

# %% id="GcIn4i9CUpb7" outputId="9475ba33-d43d-4f9b-927a-48aef06098c0"
# Vericamos las estadísticas de las columnas numéricas del dataframe df_aux.
df_aux.describe().T

# %% [markdown] id="xZ_FYe0uUpb7"
# Una vez que confirmamos que el dataframe *df_aux* tiene los cambios esperados, procedemos a utilizarlo para mejor la limpieza de los datos.

# %% id="KAwQk1boUpb7"
# Creamos una copia basada en df_aux con la que vamos a realizar las mejoras.
df_fix = df_aux.copy()

# %% [markdown] id="_HSsqrsfUpb7"
# Comencemos analizando qué columnas pueden estar relacionadas con las columnas que vamos a mejorar en especial las de puntaje (recordemos que las de conteo están muy relacionadas al puntaje). Es muy probable que el puntaje que se le da a un videojuego esté relacionado con valores categóricos como: a) la plataforma en la que funciona el juego (un juego puede funcionar bien en determinadas plataformas y en otras puede que no), b) el género del juego y c) la casa desarrolladora del juego (fama de la casa desarroladora. E.g. hay casas desarroladoras que tienen fama de implementar juegos con muchos bugs); y con valores numéricos como el número de ventas (pueden ser las globales). La relación que existe entre valores numéricos se puede verificar usando tablas de **correlación**. Creemos un dataframe con las columnas nombradas y las columnas a mejorar.

# %% id="ZRMcSMyYUpb7" outputId="3d928c73-31b0-4d98-8703-90e7a0301622"
# Creamos un nuevo dataframe con las columnas que se quieren analizar.
selected_columns = df_fix[[
    'Platform',
    'Genre',
    'Global_Sales',
    'Critic_Score',
    'Critic_Count',
    'User_Score',
    'User_Count',
    'Developer'
]]
selected_columns.head(50) # Imprimimos los primeros cincuenta registros.

# %% [markdown] id="K84JE_pkUpb8"
# Viendo estos primeros cincuenta elementos podemos vislumbrar cierta relación entre los puntajes, el género y, en menor medida, la casa desarrolladora. Podemos descartar una relación entre los puntajes y la plataforma pues para la misma plataforma hay puntajes muy distintos. Por otro lado los puntajes de la crítica y los puntajes de los usuarios parace estar muy parejo entre sí, y ninguno de los dos pareciera relacionarse con el número de ventas. Confirmemos esto último con una tabla de correlación.

# %% id="YVKn0_HbUpb8" outputId="f37561ee-5893-4d65-c977-da4cd7eb866e"
# Construimos un dataframe con las columnas numéricas que deseamos correlacionar
numeric_columns = selected_columns[['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']]
numeric_columns.corr()

# %% [markdown] id="W-O66ZNwUpb8"
# Esta tabla de correlación nos está mostrando una relación muy pequeña entre los datos de las columnas a mejorar y la columna *Global_Sales*. Después de este análisis pareciera que lo mejor es tratar de obtener la distribución de datos de cada columna *Critic_Score*, *Critic_Count*, *User_Score* y *User_Count* por cada género de videojuego. Comencemos por tener un conjunto de los valores de género disponibles.

# %% id="wLUQOLUoUpb8" outputId="cb0f1c67-d3a3-46bb-9d61-e0c6d688b370"
# Usamos el método unique() para obtener los valores únicos de la columna.
genres = selected_columns['Genre'].unique()
genres

# %% [markdown] id="Zi2wkdt2Upb9"
# Realicemos ahora una prueba con uno de los géneros para ver las distribuciones de, por ejemplo, la columna *Critic_Score*.

# %% id="Jxdj0g3pUpb-" outputId="1b50d62e-f7e5-496c-8be5-706c2e35caac"
# Elegimos los datos de la columna deseada filtrándolos por el género.
genre_records = selected_columns.loc[selected_columns['Genre'] == 'Action'].dropna()

# Verificamos la distribución por medio de una gráfica de histograma.
plt.hist(genre_records['Critic_Score'])
plt.show()

# %% [markdown] id="Y4tabmEOUpb-"
# Esta prueba pareciera dar una forma de campana mucho más definida, pero al estar un poco desviada seguimos optando por la mediana. Si hacemos pruebas cambiando tanto los géneros como las columnas los resultados van a ser muy parecidos. En ese sentido podemos probar por calcular la mediana de cada columna por género y tratar de imputar los valores nulos de cada columna con la mediana, también por género. Esto se debe hacer sobre el dataframe *df_fix*.

# %% id="_C6RcWzXUpb-" outputId="fb046c2a-e2cd-4a29-9054-cb77c9e80d49"
for i in range(len(genres)):
    genre = genres[i]
    g_condition = df_fix['Genre'] == genre

    # Obtenemos los registros pertenecientes a un género.
    genre_dataset = df_fix.loc[g_condition]

    # Calcular la mediana de cada columna para el género seleccionado.
    gcs_median = genre_dataset['Critic_Score'].median()
    gcc_median = genre_dataset['Critic_Count'].median()
    gus_median = genre_dataset['User_Score'].median()
    guc_median = genre_dataset['User_Count'].median()

    # Imputar los valores nulos de cada columna para el género seleccionado.
    df_fix.loc[g_condition, 'Critic_Score'] = df_fix['Critic_Score'].fillna(gcs_median)
    df_fix.loc[g_condition, 'Critic_Count'] = df_fix['Critic_Count'].fillna(gcc_median)
    df_fix.loc[g_condition, 'User_Score'] = df_fix['User_Score'].fillna(gus_median)
    df_fix.loc[g_condition, 'User_Count'] = df_fix['User_Count'].fillna(guc_median)

df_fix.isna().sum()

# %% [markdown] id="vvZgVepnUpb_"
# Hemos conseguido nuevamente reemplazar todos los valores nulos del dataset. Veamos ahora los datos estadísticos de las columnas numéricas.

# %% id="99nd0IF5Upb_" outputId="89b6aff4-5581-4487-9b9b-c480b0f6050f"
# Usamos el método describe para obtener datos estadísticos de las columnas numéricas del dataframe.
df_fix.describe().T

# %% [markdown] id="w-9y_P12Upb_"
# Estos resultados parecen tener un poco más sentido. Visualicemos las columnas mejoradas en gráficas de caja y bigotes.

# %% id="Lao-wB9SUpb_" outputId="0d162291-63ea-4a6c-cdbe-c9b03b7fe0fe"
# Visualizamos las columnas usando una gráfica de cajas y bigotes.
plt.boxplot([df_fix['Critic_Score'], df_fix['Critic_Count'], df_fix['User_Score'], df_fix['User_Count']])
plt.show()

# %% [markdown] id="OPHQrjK_UpcA"
# Podemos observar una mejora con respecto a la gráfica anterior donde prácticamente no existían cajas ni bigotes. Aun así seguimos viendo valores atípicos. Esto ocurre porque en realidad no hicimos manejo de los valores atípicos que habíamos detectado con la gráfica anterior. Lo que hicimos fue ajustar los datos, en especial los valores nulos, para poder ver con más claridad el rango de distribución de los datos. Si analizamos de manera detallada, los valores atípicos que vemos aquí son prácticamente los mismos que vimos anteriormente. Esto es algo bueno, pues significa que al hacer la imputación de los datos no generamos valores atípicos nuevos (Esto también se puede observar en la tabla de estadísticas de las columnas numéricas). En este punto el dataset está prácticamente limpio.
#
# **Nota**: al comparar con el dataset original, sí se generaron algunos valores atípicos pequeños, lo que es normal al momento de realizar imputación de datos. A pesar de esto hubo mejoras también con respecto dicho dataset (ver el caso de la columna *User_Count*).

# %% [markdown] id="P8CPvx2FUpcA"
# ### Ajustes finales
#
# Todas las columnas de tipo númerico son específicmente del tipo float, pero si analizamos no todas tienen que ser de dicho tipo. Algunas columnas como *Critic_Count* y *User_Count* que representan conteos podrían ser del tipo entero, que ocupa menos espacio en memoria y disco duro del dispositivo. Por otro lado la columna *Year_of_Release* debería ser de tipo categórico, por lo que la podríamos tranformar a tipo object (string o texto). Pero antes podríamos pasarla al tipo entero también para remover los decimales de los años. Hagamos todas estas transformaciones.

# %% id="igYOsGEEUpcA" outputId="fa166ae2-8917-4d93-99f4-98d1515a8a05"
# Aplicamos el cambio usando el método apply() y en el caso en el que se debe transformar de float a int utilizamos el tipo
# int64 perteneciente a numpy.
df_fix['Critic_Count'] = df_fix['Critic_Count'].apply(np.int64)
df_fix['User_Count'] = df_fix['User_Count'].apply(np.int64)
df_fix['Year_of_Release'] = df_fix['Year_of_Release'].apply(np.int64)

# Aplicamos el método apply() junto al tipo de dato str para pasar la columna de float a string.
df_fix['Year_of_Release'] = df_fix['Year_of_Release'].apply(str)

# Verificamos los tipos de las columnas.
df_fix.info()

# %% [markdown] id="AokMRslnUpcA"
# ### Exportar dataset
#
# Por último, cuando ya terminemos el proceso de análisis y limpieza sobre el dataset podemos proceder a exportarlo para usarlo en futuros proyectos de análisis de datos, visualización de datos o Machine Learning.

# %% id="xtBis1aVUpcB"
# Usamos el método to_csv() para exportar el dataset.
df_fix.to_csv('videogame_sales_with_ratings_cleaned.csv', index=False)
