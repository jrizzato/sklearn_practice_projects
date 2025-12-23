# Goal: to clasify a wine in one of three categories: 0, 1 or 2 depending 13 chemical characteristics

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# cargamos el dataset
data = load_wine()
print(type(data))
df = pd.DataFrame(data.data, columns=data.feature_names)


# analizamos el dataset
print("df.head():")
print(df.head(), "\n")
print("df.shape;")
print(df.shape, "\n")

df["target"] = data.target # 0, 1 o 2

print("df.head():")
print(df.head(100), "\n")
print("df.shape;")
print(df.shape, "\n")


# dividimos en datos de entrenamiento y de prueba

X = df.drop('target', axis=1) # axis=1 es una columna, axis=0 sería una fila
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape} \n")


# normalizamos los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# entrenamos, testeamos y evaluamos
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"presicion: {accuracy} \n")
print(classification_report(y_test, y_pred, target_names=data.target_names))
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusion: ")
print(cm)


# visualizacion
# plt.figure(figsize=(14,5))

# plt.subplot(1,2,1)

# plt.subplot(1,2,2)


# ejemplo de uso
nuevo_vino = {
    "alcohol": 13.2,
    "malic_acid": 1.8,
    "ash": 2.4,
    "alcalinity_of_ash": 18.5,
    "magnesium": 100.0,
    "total_phenols": 2.2,
    "flavanoids": 2.1,
    "nonflavanoid_phenols": 0.3,
    "proanthocyanins": 1.7,
    "color_intensity": 5.5,
    "hue": 1.0,
    "od280/od315_of_diluted_wines": 3.0,
    "proline": 1100.0,
}

# convertir a DataFrame respetando el orden de columnas
sample_df = pd.DataFrame([nuevo_vino])[data.feature_names]

# escalar con el mismo scaler ajustado
sample_scaled = scaler.transform(sample_df)

# predecir
pred_class = clf.predict(sample_scaled)[0]
pred_proba = clf.predict_proba(sample_scaled)[0]

print(f"Clase predicha: {pred_class} ({data.target_names[pred_class]})")
print("Probabilidades por clase:", pred_proba)