# California Housing Dataset - Regresión Lineal

## Descripción del Dataset

El dataset California Housing contiene información sobre viviendas en California basada en el censo de 1990. Se utiliza para predecir el **precio medio de las casas** en diferentes bloques geográficos.

---

## Características del Dataset

### **Tipo de Problema**
- **Regresión** (predicción de valores continuos)
- Variable objetivo: Precio de viviendas

### **Dimensiones**
- **20,640 muestras** (observaciones)
- **8 características** (features)
- **1 variable objetivo** (target)

---

## Variable Objetivo (Target)

- **PRECIO**: Precio medio de las casas (en $100,000s)
- **Rango típico**: 0.15 a 5.0 (aproximadamente $15,000 a $500,000)

---

## Correlaciones Importantes

| Característica | Correlación con Precio | Descripción |
|----------------|------------------------|-------------|
| **MedInc** | Alta positiva (~0.68) | El ingreso es el mejor predictor del precio |
| **Latitude** | Media | Ubicación norte-sur afecta el precio |
| **Longitude** | Media | Ubicación este-oeste afecta el precio |
| **HouseAge** | Baja | Edad de la casa tiene efecto limitado |
| **AveOccup** | Muy baja | Ocupación promedio poco correlacionada |

---

## Características Relevantes para Modelado

### Consideraciones
- **Contiene outliers** (especialmente en AveRooms, AveBedrms)
- **Relaciones no perfectamente lineales**
- **Datos históricos** (1990) - pueden no reflejar precios actuales
- **Datos agregados** - no son viviendas individuales

---

## Técnicas de Machine Learning Aplicables

### Implementadas en este proyecto
- **Regresión Lineal** - Baseline simple y efectivo

### Técnicas Recomendadas

#### Modelos Lineales
- **Ridge Regression** - Regularización L2 para reducir overfitting
- **Lasso Regression** - Regularización L1 y selección de features
- **ElasticNet** - Combinación de Ridge y Lasso

#### Modelos Basados en Árboles
- **Decision Tree Regressor** - Captura relaciones no lineales
- **Random Forest** - Ensemble de árboles, robusto a outliers
- **Gradient Boosting** - Alta precisión (XGBoost, LightGBM, CatBoost)

#### Otros Modelos
- **Support Vector Regression (SVR)** - Efectivo en espacios de alta dimensión
- **K-Nearest Neighbors (KNN)** - Basado en similitud geográfica
- **Neural Networks** - Para relaciones complejas

---

## Cuándo Usar Estas Técnicas

Aplica técnicas similares cuando encuentres datasets con:

- 🔹 **Problema de regresión** (predicción de valores continuos)
- 🔹 **Features numéricas** en diferentes escalas
- 🔹 **Miles de observaciones** (dataset mediano a grande)
- 🔹 **Relaciones no perfectamente lineales**
- 🔹 **Datos geográficos** (coordenadas lat/lon)
- 🔹 **Datos agregados** por grupos, bloques o regiones
- 🔹 **Presencia de outliers** que deben manejarse

---

## Métricas de Evaluación

Para este problema de regresión, las métricas relevantes son:

| Métrica | Descripción | Interpretación |
|---------|-------------|----------------|
| **R² Score** | Proporción de varianza explicada | 0-1, mayor es mejor |
| **MSE** | Error cuadrático medio | Menor es mejor |
| **RMSE** | Raíz del MSE (mismas unidades) | Menor es mejor |
| **MAE** | Error absoluto medio | Menor es mejor, robusto a outliers |

---

## Resultados del Modelo

### Regresión Lineal
- **R² Score**: ~0.60 (explica 60% de la varianza)
- **MSE**: ~0.52
- **RMSE**: ~0.72 ($72,000 de error promedio)

---

## Mejoras Potenciales

1. **Ingeniería de Features**
   - Crear interacciones (ej: MedInc × Latitude)
   - Features polinómicas
   - Transformaciones logarítmicas

2. **Preprocesamiento**
   - Normalización/Estandarización de features
   - Manejo de outliers
   - Feature selection

3. **Modelos Avanzados**
   - Probar Random Forest o Gradient Boosting
   - Hyperparameter tuning con GridSearchCV
   - Ensemble methods

4. **Validación**
   - Cross-validation
   - Validación en datos de diferentes años

---

## 📅 Información del Proyecto

- **Fecha de creación**: 13 de diciembre de 2025
- **Lenguaje**: Python
- **Librerías**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Modelo**: Regresión Lineal
