# Yeni bir görüntüyü yükleme ve ön işleme
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('C:/Users/yemre/OneDrive/Masaüstü/chest_xray/chest_xray/train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_datagen = ImageDataGenerator(rescale= 1./255)
test_set = test_datagen.flow_from_directory('C:/Users/yemre/OneDrive/Masaüstü/chest_xray/chest_xray/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

test_image = image.load_img('C:/Users/yemre/OneDrive/Masaüstü/data/sanana/hasta10.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)



import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

# Eğitilmiş modeli yükleme
cnn = load_model('chest_xray_model.h5')

# Modelin giriş boyutunu kontrol etme (isteğe bağlı)
input_shape = cnn.input_shape
print(f"Model input shape: {input_shape}")
print(f"Test image shape: {test_image.shape}")

# Tahmin yapma
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Hasta'
else:
    prediction = 'Normal'
print(prediction)

import matplotlib.pyplot as plt

# Eğitim ve doğrulama metriklerini history nesnesinden alın
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



