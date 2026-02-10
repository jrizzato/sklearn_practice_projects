# 🧪 Sklearn Practice Projects

Repositorio de proyectos de práctica de Machine Learning utilizando scikit-learn, organizados por tipo de algoritmo.

## 📊 Estructura del Repositorio

Este repositorio contiene proyectos demostrativos organizados en tres categorías principales:

- **Classification** (Clasificación)
- **Clustering** (Agrupamiento)
- **Regression** (Regresión)

---

## 🎯 Proyectos de Clasificación

### 1. **Iris Flower Classification** (`/classification/iris/`)
Clasificación de flores Iris en tres especies basándose en características físicas.

**Archivos:**
- `iris.py` - Clasificación con Regresión Logística
- `iris_knn.py` - Clasificación con K-Nearest Neighbors (KNN)
- `iris_SVM.py` - Clasificación con Support Vector Machine + GridSearchCV
- `iris_RFC.py` - Clasificación con Random Forest
- `iris_fastapi_app.py` - API REST con FastAPI para inferencia del modelo

**Dataset:** Scikit-learn Iris Dataset  
**Algoritmos:** Logistic Regression, KNN, SVM, Random Forest  
**Características:** 4 features (sepal/petal length/width)  
**Clases:** 3 especies de iris

---

### 2. **Wine Classification** (`/classification/wine/`)
Clasificación de vinos en tres categorías según 13 características químicas.

**Archivo:** `wine.py`  
**Dataset:** Scikit-learn Wine Dataset  
**Algoritmo:** Logistic Regression  
**Características:** 13 propiedades químicas  
**Clases:** 3 tipos de vino (0, 1, 2)

---

### 3. **Credit Risk Prediction** (`/classification/credit_risk/`)
Predicción de riesgo crediticio (incumplimiento de préstamos).

**Archivo:** `credit_risk.py`  
**Dataset:** Simulado (21 registros)  
**Algoritmo:** Decision Tree Classifier  
**Target:** Defaulted (0: riesgo bajo, 1: riesgo alto)  
**Features:** Age, Income, LoanAmount, CreditHistory

📄 [Ver README detallado](classification/credit_risk/readme.md)

---

### 4. **Breast Cancer Classification** (`/classification/breast_cancer/`)
Clasificación de tumores de mama como malignos o benignos.

**Archivo:** `breast_cancer.py`  
**Dataset:** Scikit-learn Breast Cancer Dataset  
**Algoritmo:** Random Forest Classifier  
**Target:** Tumor type (0: maligno, 1: benigno)  
**Características:** 30 features médicos  
**Visualizaciones:** Feature importance, confusion matrix, árboles individuales

---

### 5. **Handwritten Digits Recognition** (`/classification/digits/`)
Reconocimiento de dígitos escritos a mano (0-9).

**Archivo:** `digits.py`  
**Dataset:** Scikit-learn Digits Dataset (1,797 imágenes de 8x8 pixels)  
**Algoritmo:** Support Vector Machine (SVM) con kernel RBF  
**Target:** Dígito reconocido (0-9)  
**Features:** 64 valores de píxeles por imagen

---

## 🔵 Proyectos de Clustering

### 1. **Customer Segmentation** (`/clustering/customer_segmentation/`)
Segmentación de clientes basada en ingresos anuales y puntuación de gasto.

**Archivo:** `customer_segmentation.py`  
**Dataset:** `Mall_Customers.csv` (67 clientes)  
**Algoritmo:** K-Means Clustering  
**Features:** Annual Income, Spending Score  
**Clusters:** 5 grupos (determinados por método Elbow)  
**Visualizaciones:** Elbow curve, scatter plot de clusters

---

## 📈 Proyectos de Regresión

### 1. **Titanic Survival Prediction** (`/regression/Titanic/`)
Predicción de supervivencia de pasajeros del Titanic.

**Archivo:** `Titanic.py`  
**Dataset:** Seaborn Titanic Dataset  
**Algoritmo:** Logistic Regression  
**Target:** Survived (0: murió, 1: sobrevivió)  
**Features:** pclass, sex, age, sibsp, parch, fare, embarked

📄 [Ver README detallado](regression/Titanic/readme.md)

---

### 2. **California Housing Price Prediction** (`/regression/california_housing/`)
Predicción de precios de viviendas en California.

**Archivo:** `california_housing.py`  
**Dataset:** Scikit-learn California Housing (20,640 muestras)  
**Algoritmo:** Linear Regression  
**Target:** Precio medio de casas (en $100,000s)  
**Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude  
**Métricas:** R², MSE

📄 [Ver README detallado](regression/california_housing/readme.md)

---

### 3. **Diabetes Progression Prediction** (`/regression/diabetes/`)
Predicción de progresión de diabetes usando múltiples algoritmos.

**Archivos:**
- `diabetes_LinearRegression.py` - Regresión Lineal
- `diabetes_Ridge.py` - Ridge Regression
- `diabetes_RandomForest.py` - Random Forest Regressor
- `diabetes_GradientBoosting.py` - Gradient Boosting Regressor

**Dataset:** Scikit-learn Diabetes Dataset  
**Target:** Medida cuantitativa de progresión de la enfermedad  
**Features Utilizadas:** Subset de 6 características (excluyendo age, sex, bmi, bp)  
**Métricas:** MSE, R², MAE

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Scikit-learn** - Algoritmos de ML y datasets
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Matplotlib** - Visualizaciones
- **Seaborn** - Visualizaciones estadísticas
- **FastAPI** - API REST para inferencia de modelos
- **Joblib** - Serialización de modelos

---

## 📦 Instalación

```bash
pip install pandas numpy scikit-learn matplotlib seaborn fastapi joblib
```

---

## 🚀 Uso

Cada proyecto puede ejecutarse independientemente:

```bash
# Ejemplo: ejecutar clasificación de iris
python classification/iris/iris.py

# Ejemplo: ejecutar segmentación de clientes
python clustering/customer_segmentation/customer_segmentation.py

# Ejemplo: ejecutar predicción de precios de casas
python regression/california_housing/california_housing.py
```

---

## 📚 Conceptos Cubiertos

### Algoritmos de Clasificación
- ✅ Logistic Regression
- ✅ K-Nearest Neighbors (KNN)
- ✅ Support Vector Machine (SVM)
- ✅ Decision Trees
- ✅ Random Forest

### Algoritmos de Clustering
- ✅ K-Means
- ✅ Método Elbow para optimización de k

### Algoritmos de Regresión
- ✅ Linear Regression
- ✅ Ridge Regression (Regularización L2)
- ✅ Random Forest Regressor
- ✅ Gradient Boosting Regressor

### Técnicas de ML
- ✅ Train-test split
- ✅ Cross-validation (GridSearchCV)
- ✅ Feature scaling (StandardScaler)
- ✅ Model evaluation (accuracy, precision, recall, F1, R², MSE, MAE)
- ✅ Confusion matrix
- ✅ Feature importance analysis
- ✅ Model deployment (FastAPI)

---

## 📊 Métricas de Evaluación

### Clasificación
- **Accuracy Score**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **Classification Report**

### Regresión
- **R² (Coefficient of Determination)**
- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**

### Clustering
- **Inertia**
- **Método Elbow**

---

## 📝 Notas

- Todos los datasets utilizados son públicos y están disponibles en scikit-learn o Seaborn
- Cada proyecto incluye visualizaciones para mejor comprensión
- Los modelos están configurados con `random_state` para reproducibilidad
- Se incluyen ejemplos de predicción con datos personalizados

---

## 🔗 Referencias

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Seaborn Datasets](https://github.com/mwaskom/seaborn-data)

---

## 📫 Contacto

**Autor:** jrizzato  
**GitHub:** [jrizzato/sklearn_practice_projects](https://github.com/jrizzato/sklearn_practice_projects)
