import tensorflow as tf
mnist = tf.keras.datasets.mnist

tf.config.set_visible_devices([],'GPU')

# Directorio donde se guardará el modelo
model_dir = 'mnist_model'
model_path = os.path.join(model_dir, 'model.h5')

#cargamos y procesamos los datos
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#definimos el modelo
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

#compilamos el modelo
model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

#entrenamos
model.fit(x_train, y_train, epochs=5)

#evalua
model.evaluate(x_test, y_test)