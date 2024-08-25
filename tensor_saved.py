import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

# Establece que solo se use la CPU
tf.config.set_visible_devices([], 'GPU')

# Directorio donde se guardará el modelo
project_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_dir, 'models')
model_path = os.path.join(model_dir, 'model.keras')

# Cargar y preprocesar los datos
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Verificar si el modelo ya está guardado
if os.path.exists(model_path):
    print("Modelo encontrado. Cargando modelo desde la carpeta...")
    model = tf.keras.models.load_model(model_path)
else:
    print("Modelo no encontrado. Entrenando nuevo modelo...")
    
    # Definir el modelo
    model = tf.keras.models.Sequential([
        # solo los datos estan en una matriz de 28x28 
        # y lo convierte a un vector de 784 para la sigt capa
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        #Función: Una capa densa (o fully connected) con 128 unidades (neuronas). 
        # Cada neurona en esta capa está conectada a todas las neuronas de la capa anterior. 
        # Esta capa aprende características complejas de los datos.
        #Param #: 100,480. Esto se calcula como 784 (entradas) * 128 (neuronas) + 128 (bias). 
        # Cada neurona tiene un peso para cada entrada y un bias. 
        tf.keras.layers.Dense(128, activation='relu'),

        #Función: Dropout es una técnica de regularización que desactiva aleatoriamente
        #  una fracción de las neuronas durante el entrenamiento, lo que ayuda a prevenir 
        # el sobreajuste (overfitting). En tu caso, se está aplicando a 128 neuronas.
        #Param #: 0, porque no tiene parámetros que aprender.
        tf.keras.layers.Dropout(0.2),

        #Función: Esta es la capa de salida con 10 unidades (neuronas), una para cada clase 
        # en el conjunto de datos MNIST (dígitos del 0 al 9). Utiliza la función de activación softmax 
        # para producir una probabilidad para cada clase.
        #Param #: 1,290. Esto se calcula como 128 (entradas) * 10 (salidas) + 10 (bias)
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Entrenar el modelo
    model.fit(x_train, y_train, epochs=5)
    
    # Crear el directorio si no existe y guardar el modelo
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

model.summary()

# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Precisión del modelo: {accuracy}")

new_image = x_test[8]

new_image_expanded = np.expand_dims(new_image, axis=0)
predictions = model.predict(new_image_expanded)
#print(f"arrray {predictions}")
for i, prob in enumerate(predictions[0]):
    print(f"Clase {i}: {prob:.4f}")
predicted_class = np.argmax(predictions)

plt.imshow(new_image, cmap='gray')
plt.title(f"Predicción: {predicted_class}")
plt.axis('off')
plt.show()

print(f"Predicción: {predicted_class}")

