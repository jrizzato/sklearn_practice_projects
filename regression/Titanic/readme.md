# Predicción de Supervivencia del Titanic

Proyecto de machine learning que predice si un pasajero del Titanic sobrevivió o no, utilizando regresión logística.

Regresión Logística, es un algoritmo de clasificación. Puede ser binaria, multimodal u ordinal.

## Descripción

Este proyecto utiliza el dataset del Titanic de Seaborn para entrenar un modelo de clasificación binaria que predice la supervivencia de los pasajeros basándose en características como clase, sexo, edad, etc.

## Requisitos

```bash
pip install pandas seaborn scikit-learn matplotlib
```

## Variable Objetivo

- `survived`: Variable binaria que indica si el pasajero sobrevivió
  - **0** = Murió
  - **1** = Sobrevivió

## Características (Features)

- `pclass`: Clase del pasajero (1, 2, 3)
- `sex`: Sexo (0=masculino, 1=femenino)
- `age`: Edad del pasajero
- `sibsp`: Número de hermanos/cónyuges a bordo
- `parch`: Número de padres/hijos a bordo
- `fare`: Tarifa pagada por el pasajero
- `embarked`: Puerto de embarque (0=Southampton, 1=Cherbourg, 2=Queenstown)

## Preprocesamiento de Datos

- Eliminación de columnas irrelevantes o con muchos datos faltantes
- Eliminación de filas con valores nulos (NaN)
- Codificación de variables categóricas (`sex` y `embarked`)
- División de datos: 90% entrenamiento, 10% prueba

## Uso

Ejecuta el script principal:

```bash
python main.py
```

El script mostrará:
- Accuracy del modelo
- Reporte de clasificación (precision, recall, f1-score)
- Matriz de confusión (visualización con heatmap)
- Predicción de ejemplo para un pasajero personalizado

## Modelo

- **Algoritmo**: Regresión Logística
- **Parámetros**: `max_iter=1000`
- **Métrica principal**: Accuracy