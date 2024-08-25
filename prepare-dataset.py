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

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Redimensiona las imágenes a 224x224
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# Extraer las imágenes y etiquetas del generador de datos
X_train, y_train = [], []
for i in range(len(train_generator)):
    images, labels = train_generator[i]
    X_train.extend(images)
    y_train.extend(labels)
X_train = np.array(X_train)
y_train = np.array(y_train)

X_val, y_val = [], []
for i in range(len(validation_generator)):
    images, labels = validation_generator[i]
    X_val.extend(images)
    y_val.extend(labels)
X_val = np.array(X_val)
y_val = np.array(y_val)

# Guardar los arrays en archivos .npz
np.savez('train_data.npz', X_train=X_train, y_train=y_train)
np.savez('val_data.npz', X_val=X_val, y_val=y_val)

print("Datasets guardados como archivos .npz")
