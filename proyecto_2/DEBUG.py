# %% [markdown]
# ![](https://www.dii.uchile.cl/wp-content/uploads/2021/06/Magi%CC%81ster-en-Ciencia-de-Datos.png)
# 

# %% [markdown]
# # Proyecto: Riesgo en el Banco Giturra
# 
# **MDS7202: Laboratorio de Programación Científica para Ciencia de Datos**
# 
# ### Cuerpo Docente:
# 
# - Profesor: Gabriel Iturra, Ignacio Meza De La Jara
# - Auxiliar: Sebastián Tinoco
# - Ayudante: Arturo Lazcano, Angelo Muñoz
# 
# _Por favor, lean detalladamente las instrucciones de la tarea antes de empezar a escribir._
# 
# ---
# 
# ## Reglas
# 
# - Fecha de entrega: 19/12/2023
# - **Grupos de 2 personas.**
# - Cualquier duda fuera del horario de clases al foro. Mensajes al equipo docente serán respondidos por este medio.
# - Estrictamente prohibida la copia.
# - Pueden usar cualquier material del curso que estimen conveniente.
# 

# %% [markdown]
# ---
# 

# %% [markdown]
# # Presentación del Problema
# 

# %% [markdown]
# ![](https://www.diarioeldia.cl/u/fotografias/fotosnoticias/2019/11/8/67218.jpg)
# 

# %% [markdown]
# **Giturra**, un banquero astuto y ambicioso, estableció su propio banco con el objetivo de obtener enormes ganancias. Sin embargo, su reputación se vio empañada debido a las tasas de interés usureras que imponía a sus clientes. A medida que su banco crecía, Giturra enfrentaba una creciente cantidad de préstamos impagados, lo que amenazaba su negocio y su prestigio.
# 
# Para abordar este desafío, Giturra reconoció la necesidad de reducir los riesgos de préstamo y mejorar la calidad de los préstamos otorgados. Decidió aprovechar la ciencia de datos y el análisis de riesgo crediticio. Contrató a un equipo de expertos para desarrollar un modelo predictivo de riesgo crediticio.
# 
# Cabe señalar que lo modelos solicitados por el banquero deben ser interpretables. Ya que estos le permitira al equipo comprender y explicar cómo se toman las decisiones crediticias. Utilizando visualizaciones claras y explicaciones detalladas, pudieron identificar las características más relevantes, le permitirá analizar la distribución de la importancia de las variables y evaluar si los modelos son coherentes con el negocio.
# 
# Para esto Giturra les solicita crear un modelo de riesgo disponibilizandoles una amplia gama de variables de sus usuarios: como historiales de crédito, ingresos y otros factores financieros relevantes, para evaluar la probabilidad de incumplimiento de pago de los clientes. Con esta información, Giturra podra tomar decisiones más informadas en cuanto a los préstamos, ofreciendo condiciones más favorables a aquellos con menor riesgo de impago.
# 

# %% [markdown]
# ## Instalación de Librerías y Carga de Datos.
# 

# %% [markdown]
# Para el desarrollo de su proyecto, utilice el conjunto de datos `dataset.pq` para entrenar un modelo de su elección. Se le recomienda levantar un ambiente de `conda` para instalar las librerías y así evitar cualquier problema con las versiones.
# 

# %% [markdown]
# ---
# 
# ## Secciones Requeridas en el Informe
# 
# La siguiente lista detalla las secciones que debe contener su notebook para resolver el proyecto. 
# Es importante que al momento de desarrollar cada una de las secciones, estas sean escritas en un formato tipo **informe**, donde describan detalladamente cada uno de los puntos realizados.
# 
# ### 1. Introducción [0.5 puntos]
# 
# _Esta sección es literalmente una muy breve introducción con todo lo necesario para entender que hicieron en su proyecto._
# 
# - Describir brevemente el problema planteado (¿Qué se intenta predecir?)
# - Describir brevemente los datos de entrada que les provee el problema.
# - Describir las métricas que utilizarán para evaluar los modelos generados. Eligan **una métrica** adecuada para el desarrollo del proyecto **según la tarea que deben resolver y la institución a la cuál será su contraparte** y luego justifiquen su elección. Considerando que los datos presentan desbalanceo y que el uso de la métrica 'accuracy' sería incorrecto, enfoquen su elección en una de las métricas precision, recall o f1-score y en la clase que será evaluada.
# - [Escribir al final] Describir brevemente el modelo que usaron para resolver el problema (incluyendo las transformaciones intermedias de datos).
# - [Escribir al final] Indicar si lograron resolver el problema a través de su modelo. Indiquen además si creen que los resultados de su mejor modelo son aceptables y como les fue con respecto al resto de los equipos.
# 
# ### 2. Carga de datos Análisis Exploratorio de Datos [Sin puntaje]
# 
# _La idea de esta sección es que cargen y exploren el dataset para así obtener una idea de como son los datos y como se relacionan con el problema._
# 
# Cargue los datos y realice un análisis exploratorio de datos para investigar patrones, tendencias y relaciones en un conjunto de datos. Se adjuntan diversos scripts para abodar rápidamente este punto. La descripción de las columnas las pueden encontrar en el siguiente [enlace](https://www.kaggle.com/datasets/parisrohan/credit-score-classification).
# 
# **NO deben escribir nada**, solo ejecutar el código y encontrar los patrones con los cuales se basaran para generar el modelo.
# 
# ### 3. Preparación de Datos [0.5 puntos]
# 
# _Esta sección consiste en generar los distintos pasos para preparar sus datos con el fin de luego poder crear su modelo._
# 
# #### 3.1 Preprocesamiento con `ColumnTransformer`
# 
# - Convierta las columnas mal leidas a sus tipos correspondientes (float, str, etc...)
# - Genere un `ColumnTransformer` que:
#   - Preprocese datos categóricos y ordinales.
#   - Escale/estandarice datos numéricos.
#   - Uitlice `.set_output(transform="pandas")` sobre su `ColumnTransformer` para setear el formato de salida a de las transformaciones a pandas.
# 
# - Luego, pruebe las transformaciones utilizando `fit_transform`.
# 
# - Posteriormente, ejecute un Holdout que le permita más adelante evaluar los modelos.
# 
# #### 3.2 Holdout 
# 
# Ejecute `train_test_split` para generar un conjunto de entrenamiento, validacióny de prueba. 
# 
# #### 3.3 Datos nulos.
# 
# Como habrá visto, existe la posibilidad de que algunos datos sean nulos. En esta sección se le solicita justificar, previo a comenzar el modelado, decidir si conservar e imputar los datos nulos o eliminar las filas. 
# 
# Note que la decisión que tomen aquí puede afectar fuertemente el rendimiento de los modelos. 
# Y como siempre, más adelante tienen el espacio para experimentar con ambas opciones.
# 
# #### 3.4 Feature Engineering [Bonus - 0.5 puntos]
# 
# En esta sección, se espera que apliquen su conocimiento y creatividad para identificar y construir características que brinden una mejor orientación a su modelo para identificar los casos deseados. Para motivar la construcción de nuevas características, se recomienda explorar las siguientes posibilidades:
# 
# - Generar ratios que relacionen variables categóricas con numéricas. Estos ratios permiten capturar relaciones proporcionales o comparativas entre diferentes categorías y valores numéricos.
# - Combinación de rankings entre variables numéricas y categóricas.
# - Discretización de variables numéricas a categóricas.
# - Etc...
# 
# **Importantes**: Al explorar estas posibilidades no se limiten solo a estas propuestas, pueden aplicar otras técnicas de feature engineering pertinentes para mejorar la capacidad de su modelo para comprender y aprovechar los patrones presentes en los datos. 
# 
# ### 4. Baseline [1.5 puntos]
# 
# _En esta sección deben crear los modelos más básicos posibles que resuelvan el problema dado. La idea de estos modelos son usarlos como comparación para que en el siguiente paso lo puedan mejorar._
# 
# Implemente, entrene y evalúe varias `Pipeline` enfocadas en resolver el problema de clasificación en donde la diferencia entre estas sea el modelo utilizado.
# 
# 
# Para esto, cada Pipeline debe:
# 
# - Tener el `ColumnTransformer` implementado en la sección anterior como primer paso.
# - Implementar un imputador en caso de haber decidido conservar los datos nulos.
# - Implementar un clasificador en la salida (ver siguiente lista).
#   
# Y además, 
# - Ser evaluado de forma general imprimiendo un `classification_report`.
# - Calcular y guardar la métrica seleccionada en el punto 1.2 en un arreglo de métricas (guardar nombre y valor de la métrica).
# 
# Lo anterior debe ser implementado utilizando los siguientes modelos:
# 
# - `Dummy` con estrategia estratificada.
# - `LogisticRegression`.
# - `KNeighborsClassifier`.
# - `DecisionTreeClassifier`
# - `SVC`
# - `RandomForestClassifier` 
# - `LightGBMClassifier` (del paquete `lightgbm`)
# - `XGBClassifier` (del paquete `xgboost`).
# 
# 
# Luego, transformando el diccionario de las métricas a un pandas `DataFrame`, ordene según los valores de su métrica de mayor a menor y responda.
# - ¿Hay algún clasificador entrenado mejor que el azar (`Dummy`)?
# - ¿Cuál es el mejor clasificador entrenado?
# - ¿Por qué el mejor clasificador es mejor que los otros?
# - Respecto al tiempo de entrenamiento, con cual cree que sería mejor experimentar (piense en el tiempo que le tomaría pasar el modelo por una grilla de optimización de hiperparámetros).
# 
# **Nota**: Puede utilizar un for más una lista con las clases de los modelos mencionados para simplificar el proceso anterior.
# 
# 
# ### 5. Optimización del Modelo [1.5 puntos]
# 
# _En esta sección deben mejorar del modelo de clasificación al variar los algoritmos/hiperparámetros que están ocupando._
# 
# - Instanciar dos nuevas `Pipeline`, similares a la anterior, pero ahora enfocada en buscar el mejor modelo. Para esto, la pipelines debe utilizar el primer y segundo mejor modelo encontrado en el paso anterior.
# - Usar **`Optuna`** para tunear hiperparámetros
# - **Importante**: Recuerden setear la búsqueda para optimizar la métrica seleccionada en los puntos anteriores.
# 
# Algunas ideas para mejorar el rendimiento de sus modelos:
# 
# - Agregar técnicas de seleccion de atributos/características. El parámetro de cuántas características se seleccionan debe ser parametrizable y configurado por el optimizador de hiperparámetros.
# - Variar el imputador de datos en caso de usarlo.
# 
# #### Bonus
# 
# 1. **Visualización con Optuna** [0.2 extras]: Explore la documentación de visualización de Optuna en el siguiente [link](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html) y realice un análisis sobre el proceso de optimización de hiperparámetros realizado.
# 2. **Imabalanced learn** [0.3 extras]: Al ser el problema desbalanceado, pueden probar técnicas para balancear automáticamente el dataset previo a ejecutar el modelo. Para esto, puede probar con los mecanismos implementados en la librería [Imbalanced learn](https://imbalanced-learn.org/). 
# 3. **Probar pycaret (AutoML)** [0.3 extras].
# 
# Algunas notas interesantes sobre este proceso:
# 
# - No se les pide rendimientos cercanos al 100% de la métrica para concretar exitosamente el proyecto. Por otra parte, celebren cada progreso que obtengan.
# - **Hacer grillas computables**: Si la grilla se va a demorar 1/3 la edad del universo en explorarse completamente, entonces achíquenla a algo que sepan que va a terminar.
# - Aprovechen el procesamiento paralelo (con `njobs`) para acelerar la búsqueda. Sin embargo, si tienen problemas con la memoria RAM, reduzca la cantidad de jobs a algo que su computador/interprete web pueda procesar.
# 
# **Al final de este proceso, seleccione el mejor modelo encontrado, prediga el conjunto de prueba y reporte sus resultados.**

# %% [markdown]
# ### 6. Interpretabilidad [1.0 puntos]
# 
# _En esta sección, se espera que los estudiantes demuestren su capacidad para explicar cómo sus modelos toman decisiones utilizando los datos. Dentro del análisis de interpretabilidad propuesto para el modelo, deberán ser capaces de:_
# 
# - Proponer un método para analizar la interpretabilidad del modelo. Es crucial que puedan justificar por qué el método propuesto es el más adecuado y explicar los alcances que podría tener en su aplicación.
# - Identificar las características más relevantes del modelo. ¿La distribución de importancia es coherente y equitativa entre todas las variables?
# - Analizar 10 observaciones aleatorias utilizando un método específico para verificar la coherencia de las interacciones entre las características.
# - Explorar cómo se relacionan las variables utilizando algún descriptivo de interpretabilidad.
# - ¿Existen variables irrelevantes en el problema modelado?, ¿Cuales son?.
# 
# Es fundamental que los estudiantes sean capaces de determinar si su modelo toma decisiones coherentes y evaluar el impacto que podría tener la aplicación de un modelo con esas variables en una población. ¿Es posible que el modelo sea perjudicial o que las estimaciones se basen en decisiones sesgadas?
# 
# En resumen, esta sección busca que los estudiantes apliquen un enfoque crítico para evaluar la interpretabilidad de su modelo, identificar posibles sesgos y analizar las implicaciones de sus decisiones en la población objetivo.
# 
# ### 7. Concluir [1.0 puntos]
# 
# _Aquí deben escribir una breve conclusión del trabajo que hicieron en donde incluyan (pero no se limiten) a responder las siguientes preguntas:_
# 
# - ¿Pudieron resolver exitosamente el problema?
# - ¿Son aceptables los resultados obtenidos?
# - ¿En que medida el EDA ayudó a comprender los datos en miras de generar un modelo predictivo?
# 
# Respecto a la clasificación:
# 
# - ¿Como fue el rendimiento del baseline para la clasificación?
# - ¿Pudieron optimizar el baseline para la clasificación?
# - ¿Que tanto mejoro el baseline de la clasificación con respecto a sus optimizaciones?
# 
# Finalmente:
# 
# - ¿Estuvieron conformes con sus resultados?
# - ¿Creen que hayan mejores formas de modelar el problema?
# - ¿En general, qué aprendieron del proyecto? ¿Qué no aprendieron y les gustaría haber aprendido?
# 
# **OJO** si usted decide responder parte de estas preguntas, debe redactarlas en un formato de informe y no responderlas directamente.
# 
# ### Otras Instrucciones
# 
# Recordar el uso de buenas prácticas de MLOPS como replicabilidad (fijar semillas aleatorias) o el uso del registro de experimentos (con MLFlow). Si bien son opcionales, es altamente recomendado su uso.
# 
# ### 8. Bonus: Implementación de Kedro y FastAPI [1.5 puntos]
# 
# **OPCIONAL**
# 
# En esta sección se les solicita utilizar las últimas tecnologías vistas en el curso para la productivización del proyecto de ciencia de datos, centrándose en la organización y gestión de los flujos de trabajo a través de componentes y pipelines, más el servicio del modelo a través del desarrollo de una API.
# 
# Para esto: 
# 
# 1. Genere un proyecto de `Kedro` en donde separe por responsabilidades los nodos/componentes de su proyecto de ciencia de datos en módulos separados. [1.0 puntos]
# 2. Genere un servidor basado en `FastAPI` el cuál a través de un método post, reciba un batch de datos y genere predicciones para cada uno de ellos. [0.5 puntos]
# 
# Las implementaciones son libres. Es decir, usted decide qué componentes implementar, como usar el catálogo de datos y la parametrización del flujo. Sin embargo, evaluaremos buen uso de los framework, modularización y separación de responsabilidades.
# 

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=87110296-876e-426f-b91d-aaf681223468' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>

# %% [markdown]
# # INICIO INFORME

# %%
%pip install pandas numpy IPython glob2 matplotlib numba scikit-learn scipy pyarrow
#%pip install requirements.txt 

# %%
import pandas as pd
import numpy as np
import os
import IPython
import glob
import matplotlib.pyplot as plt
import numba as nb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder,MinMaxScaler,FunctionTransformer,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier 
import xgboost as xgb
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns

# %% [markdown]
# ### 2. Análisis Exploratorio de los Datos

# %% [markdown]
# En primer lugar, se cargan los datos. Además, se visualizan los cinco primeros elementos para verificar visualmente las variables. Para terminar el análisis preliminar, se llama al método `.describe()` del dataframe para visualizar el comportamiento estadístico de los datos numéricos. 

# %%
dataset = pd.read_parquet("dataset.pq")

# %%
dataset.head()

# %%
dataset.describe().round(2)

# %% [markdown]
# Los datos provistos por el banco de Giturra corresponden al dataset "dataset.pq". Cuenta con 22 columnas y 12500 líneas. Cada fila corresponde a un cliente del banco, y las 22 columnas describen algunos de los atributos del cliente, como por ejemplo su edad, ingreso anual, saliario, número de cuentas bancarias, comportamiento de pago, balance mensual, etc.

# %%
print(dataset.dtypes)

# %%

# Set up the matplotlib figure
fig, axes = plt.subplots(9, 2, figsize=(10, 25))
fig.suptitle('Histogram Plots for DataFrame Columns', fontsize=16)

# Flatten the axes array
axes = axes.flatten()

# Loop through each column and plot histogram
for i, column in enumerate(dataset[dataset.select_dtypes(include='number').columns.to_list()]):
    sns.histplot(data=dataset[dataset.select_dtypes(include='number').columns.to_list()], x=column, ax=axes[i], kde=True, color = 'purple', bins=20)
    axes[i].set_title(f'Histogram - {column}')

fig.suptitle('Histogramas y Densidades para las Variables Numéricas del Dataset')
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
grouped_counts = dataset.groupby('credit_score').size().reset_index(name='count')
total_count = len(dataset)
grouped_counts['percentage'] = (grouped_counts['count'] / total_count) * 100
grouped_counts

# %%
correlation_matrix = dataset[dataset.select_dtypes(include='number').columns.to_list()].corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))
plt.title('Correlaciones Numéricas') 
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, mask=mask)

# Show the plot
plt.show()

# %% [markdown]
# #### Selección de variables

# %%
dataset = dataset.drop(columns = ['customer_id'])

# %% [markdown]
# ### 3. Preparación de Datos

# %% [markdown]
# Los datos se procesan por medio de  `ColumnTransformer` de la librería scikit-learn. El transformador preprocesa los datos categóricos y ordinales a través de una serie de procedimientos. 
# 

# %% [markdown]
# En primer lugar, convierte todas las columnas numéricas a tipo `float` y las columnas categóricas a `string`. 

# %%
var_objetos = dataset.select_dtypes(exclude='number').columns
var_numericas = dataset.select_dtypes(include='number').columns.to_list()
dataset[var_numericas] = dataset[var_numericas].astype(float)
for col in var_objetos:
    dataset[col] = dataset[col].astype("string")

num_X = dataset[var_numericas]
cat_X = dataset[var_objetos]

# %% [markdown]
# En segundo lugar, las columnas numéricas que incompletas (valores Nan presentes) son rellenadas con el valor promedio de la columna a la que ese valor pertenece.

# %%
dataset.isna().sum()

# %%
columns_with_nan = dataset.columns[dataset.isna().any()].tolist()
print(columns_with_nan)

# %%
dataset[columns_with_nan].describe().round(2)

# %%
num_X = num_X.fillna(num_X.mean())
data=pd.concat([num_X, cat_X], axis=1)
data = data.loc[(data[var_numericas]>=0).all(axis=1)]

# %% [markdown]
# En tercer lugar, se eliminan los outliers.

# %%
z_scores = np.abs(stats.zscore(data[var_numericas]))
df_no_outliers = data[(z_scores < 4).all(axis=1)]

# %% [markdown]
#  En cuarto lugar, se utiliza `OrdinalEncoder` para codificar las variables categóricas, y se escalan con `MinMaxScaler` las variables numéricas.

# %%
preprocessor = ColumnTransformer(transformers=[('cat', OrdinalEncoder(), var_objetos),
        ('num', MinMaxScaler(), var_numericas)])
preprocessor.set_output(transform="pandas")
X_transformed = preprocessor.fit_transform(df_no_outliers)

# %%
X_transformed

# %%
columns_with_nan = [f"num__{column}" for column in dataset.select_dtypes(include='number').columns.to_list()]

# %%

# Set up the matplotlib figure
fig, axes = plt.subplots(9, 2, figsize=(10, 25))
fig.suptitle('Histogram Plots for DataFrame Columns', fontsize=16)

# Flatten the axes array
axes = axes.flatten()

# Loop through each column and plot histogram
for i, column in enumerate(X_transformed[columns_with_nan]):
    sns.histplot(data=X_transformed[columns_with_nan], x=column, ax=axes[i], kde=True, color = 'purple', bins=20)
    axes[i].set_title(f'Histogram - {column}')

fig.suptitle('Histogramas y Densidades para las Variables Numéricas del Dataset')
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
X_transformed.describe().round(3)

# %% [markdown]
# #### Holdout

# %% [markdown]
# Lógicamente, si se esta tratando de modelar el 'credit score', para construir un modelo predictivo esta variable no puede estar disponible para el modelo. A continuación se segmentan los datos correspondientes en *X, y* (data, target) en grupos de *train*, *validation* y *test*, en proporción 40/40/20.

# %%
X = X_transformed.drop("num__credit_score",axis=1)
y=X_transformed[("num__credit_score")]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# %% [markdown]
# ### 3. Creación de Pipelines, Transformers y Modelos

# %%
dummy = DummyClassifier(strategy="stratified")
logreg = LogisticRegression()
KNC = KNeighborsClassifier()
svc = SVC()
RFC = RandomForestClassifier()
gbm = LGBMClassifier()
xgboost = xgb.XGBClassifier()

# %%
dummy_pipeline = Pipeline([('preprocessor', preprocessor),('classifier',dummy)])
logreg_pipeline = Pipeline([('preprocessor', preprocessor),('classifier',logreg)])
svc_pipeline = Pipeline([('preprocessor', preprocessor),('classifier',svc)])
rfc_pipeline = Pipeline([('preprocessor', preprocessor),('classifier',RFC)])
gbm_pipeline = Pipeline([('preprocessor', preprocessor),('classifier',gbm)])
xg_pipeline = Pipeline([('preprocessor', preprocessor),('classifier',xgboost)])


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def train(X_train, X_test, y_train, y_test,pipeline,name):
    pipeline.fit(X_train, y_train)
    display(pipeline)
    y_pred = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    print("Reporte para ",name)
    print(f'-'*60)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print("AUC:", auc)
    print(classification_report(y_test, y_pred))
    return auc



{'dummy': train(X_train,X_test,y_train,y_test,dummy_pipeline,"dummy"),
 'logreg': train(X_train,X_test,y_train,y_test,logreg_pipeline,"logreg"),
 'svc':train(X_train,X_test,y_train,y_test,svc_pipeline,"svc"),
 'rfc':train(X_train,X_test,y_train,y_test,rfc_pipeline,"RFC"),
 'gbm':train(X_train,X_test,y_train,y_test,gbm_pipeline,"LGBM"),
 'XGB':train(X_train,X_test,y_train,y_test,xg_pipeline,"XGB")}



# %% [markdown]
# 

# %% [markdown]
# 


