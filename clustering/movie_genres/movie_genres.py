# Goal: Use K-Means clustering to group movies by similar characteristics and discover genre groupings.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

# --------- creacion del dataset a partir de los archovos csv -----
# # 1. Cargar los datos
# movies = pd.read_csv('./23 - Clustering Movie Genres with K-Means/data/movies.csv')
# ratings = pd.read_csv('./23 - Clustering Movie Genres with K-Means/data/ratings.csv')

# # 2. Calcular el rating promedio por película
# avg_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
# avg_ratings.columns = ['movieId', 'avg_rating', 'num_ratings']

# # 3. Procesar los géneros (convertir a columnas binarias)
# # Primero, obtener todos los géneros únicos
# all_genres = set()
# for genres_str in movies['genres']:
#     if pd.notna(genres_str) and genres_str != '(no genres listed)':
#         all_genres.update(genres_str.split('|'))

# # Crear una columna para cada género
# genre_columns = {}
# for genre in sorted(all_genres):
#     genre_columns[genre.lower()] = movies['genres'].apply(
#         lambda x: 1 if pd.notna(x) and genre in x else 0
#     )

# # 4. Unir todo en un DataFrame
# df = pd.DataFrame({
#     'movieId': movies['movieId'],
#     'title': movies['title'],
#     **genre_columns  # Desempaquetar todas las columnas de géneros
# })

# # Agregar ratings
# df = df.merge(avg_ratings, on='movieId', how='left')

# # Rellenar valores nulos en ratings con la media
# df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())
# df['num_ratings'] = df['num_ratings'].fillna(0)

# # 5. Mostrar resultado
# print(df.head())
# print(f"\nForma del DataFrame: {df.shape}")
# print(f"\nColumnas: {df.columns.tolist()}")
#   ------- fin creacion de dataset -----

# Simulate movie data
np.random.seed(42)
n_movies = 200
df = pd.DataFrame({
'runtime': np.random.normal(100, 15, n_movies),  # in minutes
'rating': np.random.normal(6.5, 0.8, n_movies),  # IMDb-like rating
'action': np.random.randint(0, 2, n_movies),
'comedy': np.random.randint(0, 2, n_movies),
'drama': np.random.randint(0, 2, n_movies),
'romance': np.random.randint(0, 2, n_movies),
'horror': np.random.randint(0, 2, n_movies)
})
print(df.head())


# ========== VISUALIZACIONES EXPLORATORIAS ==========
# Géneros más comunes
# genre_cols = [col for col in df.columns if col not in ['movieId', 'title', 'avg_rating', 'num_ratings']]
# genre_counts = df[genre_cols].sum().sort_values(ascending=False)

# plt.figure(figsize=(12, 6))
# genre_counts.plot(kind='bar', color='steelblue', edgecolor='black')
# plt.xlabel('Género')
# plt.ylabel('Número de Películas')
# plt.title('Distribución de Géneros')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # Rating promedio por género
# genre_ratings = {}
# for genre in genre_cols:
#     movies_in_genre = df[df[genre] == 1]
#     if len(movies_in_genre) > 0:
#         genre_ratings[genre] = movies_in_genre['avg_rating'].mean()

# genre_ratings_df = pd.Series(genre_ratings).sort_values(ascending=False)

# plt.figure(figsize=(12, 6))
# genre_ratings_df.plot(kind='barh', color='coral', edgecolor='black')
# plt.xlabel('Rating Promedio')
# plt.ylabel('Género')
# plt.title('Rating Promedio por Género')
# plt.tight_layout()
# plt.show()

# # 7. Top 10 películas más populares
# top_movies = df.nlargest(10, 'num_ratings')[['title', 'avg_rating', 'num_ratings']]
# print("\nTop 10 Películas Más Populares:")
# print(top_movies)

# plt.figure(figsize=(12, 6))
# x = range(len(top_movies))
# plt.bar(x, top_movies['num_ratings'], color='skyblue', edgecolor='black')
# plt.xticks(x, top_movies['title'], rotation=45, ha='right')
# plt.ylabel('Número de Ratings')
# plt.title('Top 10 Películas Más Valoradas')
# plt.tight_layout()
# plt.show()

# df = df.drop('title', axis=1) 
# --------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters

from sklearn.decomposition import PCA
# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Add PCA to DataFrame
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]
# Scatter plot of clusters
print(df.head())
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='cluster', palette='Set2')
plt.title("Movie Clusters in 2D (PCA)")
plt.show()
