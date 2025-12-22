#  Goal: Use the Random Forest algorithm to classify breast cancer as malignant or benign based on medical features.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# #####cargamos el data set #####
data = load_breast_cancer()
print(type(data)) # <class 'sklearn.utils._bunch.Bunch'>

df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
print(df.shape)
df['target'] = data.target #agregamos 'target' al data frame (0: maligno, 1: benigno)
print(df.head())
print(df.shape)

# ##### exploramos un poco el dataset
# cantidad de cada clase
sns.countplot(x='target', data=df)
plt.title('Distribución de la clases objetivo')
plt.xticks([0,1], ["Maligno", "Benigno"])
plt.show()

# correlation heatmap
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Correlation heatmap')
plt.show()

# #### preparamos los datos para el entrenamiento
# definimos caracteristicas y variable objetivo
X = df.drop('target', axis=1) # axis=1 es una columna, axis=0 sería una fila
y = df['target']

# separamos en datos de entrenameinto y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=344)

# ##### entrenamos en random forest

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3### evaluamos el modelo

y_pred = model.predict(X_test)

#Accuracy
print('Accuracy score:', accuracy_score(y_test, y_pred)) # de sklearn.metrics

#Classification repost
print('\nClassification Report:', classification_report(y_test, y_pred)) # de sklearn.metrics

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualize most important features
importances = model.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
# Plot top 10
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Feature Importances in Breast Cancer Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Visualizar los primeros 3 árboles
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
for idx, ax in enumerate(axes):
    plot_tree(model.estimators_[idx],
              feature_names=data.feature_names,
              class_names=['Maligno', 'Benigno'],
              filled=True,
              max_depth=2,
              ax=ax)
    ax.set_title(f'Árbol {idx + 1}')
plt.tight_layout()
plt.show()
