import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

diabetes = load_diabetes()

# ver que contiene el objeto iris
print("Tipo de dato:", type(diabetes))
print("\nAtributos disponibles:", dir(diabetes))
# los que nos interesan son data, feature_names y target

print(diabetes.DESCR)

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(df.head()) # los datos ya estan normalizados!

# agregamos la columna target
df["target"] = diabetes.target
print(df.head())

# como los datos ya estan normalizados, pasamos directamente a la
# separacion en datos de entramiento y prueba
X = df.drop(['target'], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Division de datos")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# entrenamos el modelo
modelo = Ridge(alpha=0.0010)
modelo.fit(X_train, y_train)

# prediccion
y_pred = modelo.predict(X_test)
# y_pred luego se compara con y_test para ver que tan parecidos son, y_test es la realidad
# y se espera que y_pred se parezca
print("y_pred:", y_pred.shape)

# evaluamos el modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"mse:   {mse}")
print(f"R²:   {r2}")
print(f"MAE: {mae}")

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