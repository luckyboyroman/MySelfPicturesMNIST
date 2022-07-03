import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

# Чтение данных из файлов
with open("img.bin", "rb") as file1:
    x_train = pickle.load(file1)
with open("labels.bin", "rb") as file2:
    y_train = pickle.load(file2)
with open("test.bin", "rb") as file3:
    x_test = pickle.load(file3)

print("OLD: x_train SHAPE: ", x_train.shape, "y_train SHAPE: ", y_train.shape, "x_test SHAPE:", x_test.shape)
# Преобразование данных
x_train = x_train.reshape(int(x_train.shape[0] / 28), 28, 28)
x_test = x_test.reshape(int(x_test.shape[0] / 28), 28, 28)
print("NEW: x_train SHAPE: ", x_train.shape, "y_train SHAPE: ", y_train.shape, "x_test SHAPE:", x_test.shape)
# Построение модели ИНС
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(keras.optimizers.SGD(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=1)

#Предсказывание
pred = model.predict(x_test)
plt.figure()
plt.imshow(x_test[1], cmap="binary")
plt.show()
print("predict: ", np.argmax(pred[1]))
