import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# cargamos el data set
iris = load_iris()
X = iris.data
y = iris.target

# dividimos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# estandarizamos (media 0 y desvio 1) porque knn se basa en distancias
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# creamos el modelo
# elijo k=5 (se fiaj en los 5 vecinos mas cercanos)
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# evaluamos el modelo
print(f"Precisión del modelo con K={k}: {accuracy_score(y_test, y_pred):.2f}")
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ejemplo de predicción con nuevos datos
print("Ejemplo de predicción:")
ejemplo = np.array([[5.1, 3.5, 1.4, 0.2]])  # características de una muestra, es decir medidas de sépalo y pétalo
ejemplo = scaler.transform(ejemplo)
prediccion = knn.predict(ejemplo)
print("Prediccion:", prediccion)
print("tipo de dato prediccion:", type(prediccion))
probabilidades = knn.predict_proba(ejemplo)

print(f"Features: {ejemplo[0]}")
print(f"Predicción: {iris.target_names[prediccion[0]]}")
print(f"Probabilidades: {probabilidades[0]}")