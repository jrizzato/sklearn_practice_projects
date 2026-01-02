import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# simulamos un dataset de riesgo crediticio 
data = {
    "Age": [25,40,35,45,23,52,36,29,50,28, 41,33,48,30,38, 27, 31, 44, 39, 46, 34],
    "Income": [50000,60000,40000,80000,30000,90000,42000,39000,85000,76000, 58000, 62000, 70000, 54000, 61000, 55000, 57000, 63000, 68000, 59000, 64000],
    "LoanAmount": [10000, 20000, 12000, 22000, 5000, 25000, 13000, 9000, 24000, 30000, 15000, 18000, 21000, 16000, 17000, 14000, 15500, 16500, 17500, 14500, 18500],
    "CreditHistory": [1,1,0,1,0,1,0,0,1,0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    "Defaulted": [0,0,1,0,1,0,1,1,0,1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1] # 1: riesgo alto, 0:risgo bajo
}

df = pd.DataFrame(data)
print(df)

# definimos las caracteristicas y la variable objetivo
X = df[["Age", "Income", "LoanAmount", "CreditHistory"]]
y = df["Defaulted"]

# dividimos en sets de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

# entrenamos el arbol de decision
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=69)

# entrenamos el modelo
model.fit(X_train, y_train)

# hacemos una prediccion
y_pred = model.predict(X_test)

# evaluamos el modelo
print("Matriz de confusión") # de sklearn.metrics 
print(confusion_matrix(y_test, y_pred))

print("\Reporte") # de sklearn.metrics 
print(classification_report(y_test, y_pred))

# graficamos el arbol de decision
plt.figure(figsize=(12,6))
plot_tree(model, feature_names=X.columns, class_names=["riesgo bajo", "riesgo alto"]) # de sklearn.tree
plt.show()

# prediccion individual
nuevo = {"Age": 30, "Income": 60000, "LoanAmount": 15000, "CreditHistory": 1}
nuevo_df = pd.DataFrame([nuevo])
prediccion_nuevo = model.predict(nuevo_df)
print(f"La predicción para el nuevo cliente es: {'riesgo alto' if prediccion_nuevo[0]==1 else 'riesgo bajo'}")

nuevo = {"Age": 45, "Income": 75000, "LoanAmount": 9000, "CreditHistory": 5}
nuevo_df = pd.DataFrame([nuevo])
prediccion_nuevo = model.predict(nuevo_df)
print(f"La predicción para el nuevo cliente es: {'riesgo alto' if prediccion_nuevo[0]==1 else 'riesgo bajo'}")