import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os


# Directorio del proyecto
project_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio de guardado de los archivos .npz y del modelo
save_dir = os.path.join(project_dir, 'saved_data')

# Cargar el modelo entrenado
model_path = os.path.join(save_dir, 'billetes_model.keras')
model = tf.keras.models.load_model(model_path)

# Ruta de la imagen a predecir
#img_path = os.path.join(project_dir, 'data', '20', '20240310_164854_016_jpg.rf.53174e0d3ddd4f2dd7408c76ddcca1c8.jpg')  # Cambia esto a la ruta de tu imagen
#img_path = os.path.join(project_dir,'uploads', 'image_20240824_044733_358258.png')  # Cambia esto a la ruta de tu imagen
img_path = os.path.join(project_dir, 'test.png')
# Cargar y preprocesar la imagen
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  # Escalar los píxeles a [0, 1]
img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para representar un solo lote

# Mostrar la imagen que estamos prediciendo


# Realizar la predicción
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Mapear la clase predicha a la denominación correspondiente
class_indices = {0: '10', 1: '100', 2: '20'}  # Asegúrate de que esto coincida con tus clases
predicted_label = class_indices[predicted_class]

print(f"Predicción: {predicted_label}")

for i, prob in enumerate(predictions[0]):
    print(f"Clase {i}: {prob:.4f}")
predicted_class = np.argmax(predictions)

plt.imshow(img)
plt.title("Imagen a predecir")
plt.axis('off')  # Ocultar los ejes
plt.show()
