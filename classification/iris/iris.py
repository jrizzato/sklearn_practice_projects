# Goal: Use Logistic Regresion to clasify a specific type of iris flowr according to certain numeric parameters

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# cargamos el dataset
iris = load_iris()

# Ver qué contiene el objeto iris
print("Tipo de objeto:", type(iris))
print("\nAtributos disponibles:", dir(iris))

print(iris.DESCR)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# agregamos la columns target
print(df.head())
df["target"] = iris.target
print(df.head())

print("target names:", iris.target_names)

species_list = []
for valor in df["target"]:
    species_list.append(iris.target_names[valor]) # valor es 0, 1 o 2 dependiendo de la especie
df["species"] = species_list

print(df.head())
print(df.info())
print(df.describe())

# separamos en datos de prueba y entrenamiento
X = df.drop(["target", "species"], axis=1) # axis=1 indica operar sobre columnas, borra columnas en este caso
# df.drop no modifica el dataframe original
# df_sin_filas = df.drop([0, 1, 2], axis=0)  # asi sería borrar las filas 0, 1 y 2
y = df['target']

# dividimos los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\n=== DIVISIÓN DE DATOS ===")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# entrenamos el modelo
modelo = LogisticRegression(max_iter=200, random_state=42)
modelo.fit(X_train, y_train)

# prediccion
y_pred = modelo.predict(X_test)

# evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

# reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# matriz de confusión
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print()

# ejemplo de predicción con nuevos datos
print("Ejemplo de predicción:")
ejemplo = np.array([[5.1, 3.5, 1.4, 0.2]])  # características de una muestra, es decir medidas de sépalo y pétalo
ejemplo = scaler.transform(ejemplo)
prediccion = modelo.predict(ejemplo)
print("Prediccion:", prediccion)
print("tipo de dato prediccion:", type(prediccion))
probabilidades = modelo.predict_proba(ejemplo)

print(f"Features: {ejemplo[0]}")
print(f"Predicción: {iris.target_names[prediccion[0]]}")
print(f"Probabilidades: {probabilidades[0]}")
