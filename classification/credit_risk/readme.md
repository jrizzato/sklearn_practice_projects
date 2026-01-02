# Clasificador de Árbol de Decisión para Riesgo Crediticio
Proyecto de machine learning que predice el riesgo crediticio (si un cliente incumplirá o no un préstamo) utilizando un árbol de decisión.

# Descripción
Este proyecto utiliza un dataset simulado de información crediticia para entrenar un modelo de clasificación binaria que predice si un cliente representa un riesgo alto o bajo de incumplimiento.

## Requisitos
pip install pandas numpy scikit-learn matplotlib seaborn

## Variable Objetivo
Defaulted: Variable binaria que indica si el cliente incumplió el préstamo
0 = Riesgo bajo (no incumplió)
1 = Riesgo alto (incumplió)

## Características (Features)
Age: Edad del cliente
Income: Ingreso anual del cliente (en dólares)
LoanAmount: Monto del préstamo solicitado (en dólares)
CreditHistory: Historial crediticio (0=malo, 1=bueno)

## Preprocesamiento de Datos
Dataset simulado con 10 registros de ejemplo
Todas las características son numéricas (no requiere codificación)
División de datos: 70% entrenamiento, 30% prueba
No se requiere manejo de valores nulos

### Para árboles de decisión, no es necesario ni beneficioso normalizar los datos:
1 - Los árboles de decisión son invariantes a la escala: Toman decisiones basadas en umbrales (ej: "Income > 55000?"), no en distancias o magnitudes absolutas.

2 - La normalización no afecta las divisiones: Si Income > 60000 es un buen punto de corte, normalizarlo a Income_norm > 0.5 produce exactamente las mismas divisiones.

### Cuándo SÍ normalizar:
Deberías normalizar si usaras algoritmos basados en distancia o gradiente:

KNN (K-Nearest Neighbors)
SVM (Support Vector Machines)
Regresión Logística
Redes Neuronales
K-Means (clustering)

## Uso
python main.py

Ejecuta el script principal:

El script mostrará:

* Dataset completo
* Matriz de confusión
* Reporte de clasificación (precision, recall, f1-score)
* Visualización del árbol de decisión

## Modelo
* Algoritmo: Árbol de Decisión (Decision Tree Classifier)
* Parámetros:
    - criterion="entropy": Usa la ganancia de información para dividir nodos
    - max_depth=3: Profundidad máxima del árbol (evita sobreajuste)
    - random_state=69: Semilla para reproducibilidad
* Métricas principales:
* Matriz de confusión: Muestra verdaderos positivos/negativos y falsos positivos/negativos
* Precision, Recall, F1-Score: Métricas de evaluación del clasificador

## Ventajas del Árbol de Decisión
* Interpretable: Fácil de visualizar y entender las decisiones
* No lineal: Puede capturar relaciones complejas entre variables
* No requiere normalización: Funciona con datos en diferentes escalas
* Maneja características numéricas y categóricas

## Dataset
Fuente: Dataset simulado
Registros: 10 clientes de ejemplo
Características: 4 variables numéricas
Balance de clases: 50% riesgo bajo, 50% riesgo alto