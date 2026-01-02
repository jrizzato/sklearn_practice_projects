# Goal: Use GridSearchCV to tune hyperparameters for an SVM classifier on the Iris dataset and improve performance.

from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# cargamos el cada set Iris
iris = load_iris()
print(iris.DESCR)
X = iris.data
y = iris.target

print("Tipo de dato de X:", type(X))
print("Tipo de dato de y:", type(y))

print("Tamaño de X:", X.shape) # (150, 4)
print("Tamaño de y:", y.shape) # (150,)

# convertimos a dataframe para EDA
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
df["species"] = iris.target
print(df.head())

# visualizamos
sns.pairplot(df, hue="species")
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)

from sklearn.svm import SVC # support vector classification
from sklearn.model_selection import GridSearchCV

# definimos los parametros de la grilla
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"]
}

# creamos el model svm
svc = SVC()

# aplicamos gridsearchCV
#usamos gridsearch com 5 CV (cross validation)
grid = GridSearchCV(svc, param_grid, cv=5, verbose=1, scoring="accuracy", n_jobs=-1)

# ajustamos el modelo a los datos de entrenamietno
grid.fit(X_train, y_train)

# mostramos los mejores parametros y su puntuacion
print("mejores parametros:", grid.best_params_)
print("mejor CV accuracy:", grid.best_score_)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# predecimos sobre los datos de prueba
y_pred = grid.best_estimator_.predict(X_test)

#evaluamos
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

#matriz de confusion
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# ----------------------------------------------------------------
# Ejemplo de mediciones propias
import numpy as np
my_measurements = np.array([[5.1, 3.5, 1.4, 0.2]])  # Una flor
# O múltiples flores:
# my_measurements = np.array([
#     [5.1, 3.5, 1.4, 0.2],
#     [6.2, 2.9, 4.3, 1.3]
# ])

# Hacer la predicción
my_prediction = grid.best_estimator_.predict(my_measurements)

# Obtener la probabilidad de cada clase
# my_prediction_proba = grid.best_estimator_.predict_proba(my_measurements)
# Obtener la función de decisión (qué tan seguro está el modelo)
my_decision = grid.best_estimator_.decision_function(my_measurements)

# Mostrar resultado
species_names = iris.target_names
print("\n=== Predicción con mediciones propias ===")
print(f"Mediciones: {my_measurements[0]}")
print(f"Especie predicha: {species_names[my_prediction[0]]}")
print(f"Probabilidades: {dict(zip(species_names, my_decision[0]))}")