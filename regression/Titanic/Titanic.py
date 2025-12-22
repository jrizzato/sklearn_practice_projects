# Objetivo: predecir si un pasajero sobrevivió usando el dataset del titanic y logistic regresion

#  pip install pandas seaborn scikit-learn matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### preparamos los datos ####
# cargamos el data set del titanic de seabon
# datasets: https://github.com/mwaskom/seaborn-data/tree/master/raw
df = sns.load_dataset("titanic")

# imprimimos las primeras lineas
print(df.head())
print()
print(df.info())
print()

# eliminamos (drop) columnas irrelevantes o con muchos datos faltantes
df = df.drop(["deck", "embark_town", "alive", "class", "who", "adult_male", "alone"], axis=1)
# aliminamos (drop) filas con datos faltantes. hacer esto con cuidado
df = df.dropna()

# codificar (encode) las variables categóricas: sex y embarked
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q":2})

# verificamos que el dataset esté limpio
print(df.info())

#### definimos las caracteristicas y la clase objetivo ######
# features
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
# target
y = df["survived"] # 1=sobrevivió, 0=murió

### separamos en datos de entrenamiento y prueba ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=56)

### creamos y entrenamos el modelo
# creamos el modelo de regresion lineal
model = LogisticRegression(max_iter=1000)
# entrenamos el modelo
model.fit(X_train, y_train)

# ### probamos el modelo ####
# realizamos prediccion de prueba
y_pred = model.predict(X_test)
# puntuación accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nreport:\n", classification_report(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


pclass = 2
sex = 1
age = 30
sibsp = 0 # este valor es el número de hermanos / cónyuges a bordo
parch = 0 # este valor es el número de padres / hijos a bordo
fare = 15.0 # tarifa pagada por el pasajero
embarked = 1 # puerto de embarque (0=S, 1=C, 2=Q)

# prediction = model.predict([[pclass, sex, age, sibsp, parch, fare, embarked]]) # asi tira un warning
pasajero = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], 
                             columns=["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"])

prediction = model.predict(pasajero)
print(f"Predicción (0=Murió, 1=Sobrevivió): {prediction[0]}")