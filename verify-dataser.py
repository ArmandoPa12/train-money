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
