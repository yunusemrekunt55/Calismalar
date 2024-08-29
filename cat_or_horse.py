from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('C:/Users/yemre/OneDrive/Masaüstü/data/MobileNet-samples',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_datagen = ImageDataGenerator(rescale= 1./255)
test_set = test_datagen.flow_from_directory('C:/Users/yemre/OneDrive/Masaüstü/data/Yeni klasör',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=50)

import numpy as np
from keras.preprocessing import image
test_image =image.load_img('C:/Users/yemre/OneDrive/Masaüstü/data/sanana/bune.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
if test_image.dtype != np.float32:
    test_image = test_image.astype('float32')

# Modelin giriş boyutunu kontrol etme
input_shape = cnn.input_shape
print(f"Model input shape: {input_shape}")
print(f"Test image shape: {test_image.shape}")
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction ='horse'
else:
    prediction= 'cat'
print(prediction)