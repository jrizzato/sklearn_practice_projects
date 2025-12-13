# objetivo: predecir el precio de las casas en california usando el dataset de california housing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ### preparación de lso datos ###
# cargamos el dataser de california
california = fetch_california_housing()

# mostramos la descripón
print(california.DESCR)

# lo convertimos a un data frame
df = pd.DataFrame(california.data, columns=california.feature_names)
print(df.head())
print()

# le agregamos al data frame el contenido del "target" que es el precio. Es decir, le agregamso una columna más al dataframe
df["PRECIO"] = california.target
print(df.head())

# mostramos alguna gráfica
# plt.figure(figsize=(12,7))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title("Correlación de precios de casas de california (en cientos de miles)")
# plt.show()


# ### definimos las caracteristicas y la clase objetivo
# vamos a usar todas las caracteristicas (menos el precio) para determinar el precio
X = df[["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]]
# otra opción oara definir x con todas las features
# X = df.drop('PRECIO', axis=1) # creamos X con todas las features menos PRECIO
# sino, podemos elegir las features que queramos
# X = df[["HouseAge", "AveRooms", "AveBedrms"]]

y = df["PRECIO"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=512)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# inicializacion y entrenamiento del modelo de REGRESIóN LINEAL
model = LinearRegression()
model.fit(X_train, y_train)

# hacemos predicciones con los datos de prueba (test)
y_pred = model.predict(X_test)
print("y_pred:", y_pred.shape)

# calculamos y mostramos las metricas de evaluacion(test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R²:   {r2}")
print(f"MSE: {mse}")

# Scatter plot of predicted vs actual grades
# plt.figure(figsize=(8, 4))
# print(type(y_test))
# print(type(y_pred))
# print(y_test.shape)
# print(y_pred.shape)
# plt.scatter(y_test, y_pred)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--')
# plt.title("Actual vs Predicted Final Grades")
# plt.xlabel("Actual Grade")
# plt.ylabel("Predicted Grade")
# plt.grid(True)
# plt.show()

# hacer una prediccón
MedInc = 6.034
HouseAge = 63
AveRooms = 5
AveBedrms = 2
Population = 450
AveOccup = 2
Latitude = 37.88
Longitude = -122.23
y_pred = model.predict([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
print(f"Predicción: {y_pred}")

