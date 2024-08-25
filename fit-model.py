import tensorflow as tf
import numpy as np
import os

# Directorio del proyecto
project_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio de guardado de los archivos .npz
save_dir = os.path.join(project_dir, 'saved_data')

# Cargar los datos guardados
train_path = os.path.join(save_dir, 'train_data.npz')
val_path = os.path.join(save_dir, 'val_data.npz')

train_data = np.load(train_path)
X_train = train_data['X_train']
y_train = train_data['y_train']

val_data = np.load(val_path)
X_val = val_data['X_val']
y_val = val_data['y_val']

print(f"Datos de entrenamiento cargados: {X_train.shape}, {y_train.shape}")
print(f"Datos de validación cargados: {X_val.shape}, {y_val.shape}")

# Definir el modelo de red neuronal convolucional (CNN)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 clases: 10, 20, 100
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Imprimir un resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Guardar el modelo entrenado
model.save(os.path.join(save_dir, 'billetes_model.keras'))

#print("Modelo entrenado y guardado con éxito.")


# Evaluar el modelo en los datos de validación
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Precisión en el conjunto de validación: {val_accuracy:.4f}")
