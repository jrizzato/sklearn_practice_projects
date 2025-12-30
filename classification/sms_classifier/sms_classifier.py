# Goal: Build a spam detector using the Multinomial Naive Bayes algorithm on a text dataset

# vamos a usar el clásico SMS Spam Collection Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB #Multinomial Naive Bayes algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# cargamos el contenido del atchivo csv a un dataframe
# la clasificacion y el texto estan separados por una tabulacion y no tiene cabecera
df = pd.read_csv("./classification/sms_classifier/SMSSpamCollection.csv", sep="\t", header=None)
# como no tiene cabecera se las agrego
df.columns = ["label", "message"]
print(df.head())

# convertimos las etiquetas a binario: spam=1, autentico (ham)=0
df["label"] = df["label"].map({"spam": 1, "ham": 0})
print(df.head())

# verificamos las cantidades de cada uno
print(df["label"].value_counts())

x = df["message"] # texto del mensaje
y = df["label"] # spam o autentico(ham)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print()

# creamos un TF-IDF vectorizador
# TF-IDF convierte texto a números que el modelo entiende
# fit_transform: aprende vocabulario + transforma (solo en train)
# transform: solo transforma usando vocabulario aprendido (en test)
vectorizer = TfidfVectorizer(stop_words='english')

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)
print("x_train_vec:", x_train_vec.shape)
print("x_test_vec:", x_test_vec.shape)
print()

# creamos y entrenamos el modelo
model = MultinomialNB()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)
# prediccion: 1 es spam, 0 es autentico
print("y_pred:", y_pred.shape)
print()

# evaluamos el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

print("Classification report:\n", classification_report(y_test, y_pred))

def predecir_spam(mensaje):
    mensaje_vec = vectorizer.transform([mensaje])
    # print("type(mensaje):", type(mensaje))
    # print("mensaje_vec.shape:", mensaje_vec.shape)
    # print("mensaje_vec:", mensaje_vec)

    prediccion = model.predict(mensaje_vec)[0]

    return prediccion

predecir_spam_list = [
    "Congratulations! You've won a $1000 gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT! Your account will be suspended. Verify now at suspicious-link.com",
    "Thanks for your help yesterday. I really appreciate it!"
]

for mensaje in predecir_spam_list:
    prediccion = predecir_spam(mensaje)

    resultado = "SPAM" if prediccion == 1 else "HAM (auténtico)"
    print(f"Mensaje: {mensaje}")
    print(f"Predicción: {resultado}")
