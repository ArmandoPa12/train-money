import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar los datos desde el archivo JSON
with open('temperature_data.json', 'r') as file:
    data = json.load(file)['data']

# Convertir los datos en arrays de numpy
celsius = np.array([item['celsius'] for item in data])
fahrenheit = np.array([item['fahrenheit'] for item in data])

# Crear el modelo
model = models.Sequential([
    layers.Dense(units=1, input_shape=[1])  # Solo una capa con una neurona
])
# Compilar el modelo
model.compile(optimizer='sgd', loss='mean_squared_error')


# Entrenar el modelo
model.fit(celsius, fahrenheit, epochs=500, verbose=0)
# Ver los resultados del entrenamiento
print("Entrenamiento completado")


# Hacer una predicción
celsius_test = np.array([100, 0, -40], dtype=float)
predicciones = model.predict(celsius_test)
for i, pred in enumerate(predicciones):
    print(f"{celsius_test[i]}° Celsius es {pred[0]:.2f}° Fahrenheit")

