#  Goal: Use a Support Vector Machine (SVM) to classify handwritten digits using image data

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
from PIL import Image

# cargamos en data set
digits = load_digits()

# mostramos el shape de las imagenes (son 1797 muestras de digitos escritos a mano de 8x8 pixels daca una)
print("data.image.shape:", digits.images.shape)

# monsramos algunos digitos
plt.figure(figsize=(14,3))
for i in range(16):
    plt.subplot(1,16,i+1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"label: {digits.target[i]}")
    plt.axis("off")

plt.suptitle("muestras de digitos escritos a mano")
plt.show()

# "aplanamos" las imagenes (8x8 -> 64 features)
x = digits.data # cada imagen se convierte en un vector de longitud 64
y = digits.target # labels de 0-9

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=404)

#creamos la Support Vector Classifier
model = SVC(kernel='rbf', gamma=0.001, C=10)

# entrenamos el modelo con los datos de entrenaminto (x_train)
model.fit(x_train, y_train)

# evaluamos el modelo
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification report:", classification_report(y_test, y_pred))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matrix de confusion")
plt.xlabel("predicción")
plt.ylabel("realdiad")
plt.show()

# Show predictions on test images
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis('off')
plt.suptitle("SVM Predictions on Test Images")
plt.show()

def predecir_digito_propio(ruta_imagen):
    """
    Predice un dígito escrito a mano desde una imagen
    Args:
        ruta_imagen: path a la imagen (PNG, JPG, etc.)
    Returns:
        dígito predicho (0-9)
    """
    # Cargar imagen
    img = Image.open(ruta_imagen).convert('L')  # Convertir a escala de grises
    
    # Redimensionar a 8x8 píxeles
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    
    # Convertir a array numpy
    img_array = np.array(img)
    
    # Invertir colores si es necesario (el dataset usa fondo negro, dígito blanco)
    # Si tu imagen tiene fondo blanco, descomenta la siguiente línea:
    # img_array = 255 - img_array
    
    # Normalizar valores (0-16 como en el dataset original)
    img_array = (img_array / 255.0) * 16.0
    
    # Aplanar a vector de 64 elementos
    img_flat = img_array.flatten().reshape(1, -1)
    
    # Predecir
    prediccion = model.predict(img_flat)[0]
    
    # Mostrar la imagen procesada y predicción
    plt.figure(figsize=(4, 4))
    plt.imshow(img_array, cmap='gray')
    plt.title(f"Predicción: {prediccion}")
    plt.axis('off')
    plt.show()
    
    return prediccion

# Ejemplo de uso:
resultado = predecir_digito_propio("./classification/digits/unnamed.jpg")
print(f"El modelo predice: {resultado}")