from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

# Carregar dataset IRIS
df = pd.read_csv('IRIS.csv')

# Separação da classe alvo e atributos
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values 

# Transformação das classes para valores numéricos
le = LabelEncoder()
y = le.fit_transform(y)   # transforma em [0,1,2]

# One-hot encoding (necessário p/ softmax)
y = to_categorical(y)

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definição do modelo
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes

# Compilação do modelo
model.compile(optimizer=SGD(0.01),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Treinamento do modelo
H = model.fit(X_train, y_train, epochs=100, batch_size=16,
              validation_data=(X_test, y_test))

# Avaliação
loss, accuracy = model.evaluate(X_test, y_test)
print(f"[INFO] Acurácia final: {accuracy:.2f}")

# Previsões
predictions = model.predict(X_test, batch_size=1)
pred_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

class_names_pt = ["Iris-setosa", "Iris-versicolor", "Iris-virgínica"]

print("\n[INFO] Relatório de Classificação:\n")
print(classification_report(true_classes, pred_classes, target_names=class_names_pt, digits=2))

# Avaliação final
print(f"\n[INFO] Acurácia final no conjunto de teste: {accuracy*100:.2f}%")

# Gráfico de treinamento
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="Perda (treino)")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="Perda (validação)")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="Acurácia (treino)")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="Acurácia (validação)")
plt.title("Perda e Acurácia durante o Treinamento")
plt.xlabel("Época")
plt.ylabel("Perda / Acurácia")
plt.legend()
plt.show()