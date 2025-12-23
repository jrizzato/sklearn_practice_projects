# Segment customers based on annual income and spending score using the K-Means clustering algorithm.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ######## cargamos el dataset ########
df = pd.read_csv("./clustering/customer_segmentation/Mall_Customers.csv")

# verificamos el dataset
print(df.head())

# mostramos los nombres de las columnas
# print(df.columns)

# ######### preprocesamiento del dataset ##############1
# dropeamos el CustomerID porque no es relevante
df.drop("CustomerID", axis=1, inplace=True)
# o tambien se puede así:
# df = df.drop("CustomerID", axis=1) #el axis=1 significa que es una columna (0: filas, 1:columnas)
print(df.head())

# convertimos la caracteristica categórica Gender a numerica
# df["Genre"] = df["Genre"].map({"Male":0, "Female":1}) # aunque es irrelevante este feature
# print(df.info())

# seleccionamos la caracteristicas relevantes para segmentar
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X) # Standardize features to have mean=0 and std=1 -> pero el resultado parace el mismo
X_scaled = X

# usamos el método Elbow para encontrar el número óptimo de clusters
inertia = [] # inertia = suma de distancias al cuadrado del centro del cluster mas cercano

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# plot de la curva Elbow
plt.figure(figsize=(8,4))
plt.plot(range(1,11), inertia, marker="o")
plt.title("Método Elbow para la k óptima")
plt.xlabel("número de clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# según el método elbow elegimos k=5
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled) # esto le agrega al df la columna cluster
print(df.head())

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df['Annual Income (k$)'], 
    y=df['Spending Score (1-100)'], 
    hue=df['Cluster'], 
    palette='tab10'
)
plt.title('Customer Segments by K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# ## prueba personalizada
ingreso = 19000
indice_gasto = 79
evaluacion = pd.DataFrame([[ingreso, indice_gasto]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
prediccion = kmeans.predict(evaluacion)
print(f"El cliente con ingreso ${ingreso}k y gasto {indice_gasto} pertenece al Cluster: {prediccion[0]}")
