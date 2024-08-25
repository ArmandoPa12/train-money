import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorio del proyecto
project_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio de tus datos
data_dir = os.path.join(project_dir, 'data')

# Verifica la ruta de 'data_dir'
print(f"Ruta del directorio de datos: {data_dir}")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Redimensiona las imágenes a 224x224
    batch_size=32,
    class_mode='sparse',
    subset='training',
    shuffle=False  # Para que podamos guardar las etiquetas de manera coherente
)
print(train_generator.class_indices)