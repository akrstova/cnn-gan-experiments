import numpy as np
import os
from keras.preprocessing import image

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation
from keras.layers.core import Dropout
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
import datetime

dropout = 0.4
classifier = Sequential()

classifier.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3)))
classifier.add(LeakyReLU(alpha=0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(dropout))
classifier.add(Conv2D(64, (3, 3)))
classifier.add(LeakyReLU(alpha=0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(dropout))
classifier.add(Conv2D(64, (3, 3)))
classifier.add(LeakyReLU(alpha=0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(dropout))
classifier.add(Conv2D(64, (3, 3)))
classifier.add(LeakyReLU(alpha=0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(dropout))

classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=10, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'training_set/second_attempt',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'test_set/second_attempt',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical')

classifier.fit_generator(training_set, steps_per_epoch=250, epochs=25, validation_data=test_set, validation_steps=65)

# Single classifications to simulate real use case in the Android application
# product_names array stores the directory names where the test pictures will come from
# Corresponds to the one hot encoding of the classifier
product_names = ['apple', 'banana', 'cocoa', 'coffee', 'cucumber', 'onion', 'peach', 'potato', 'strawberry', 'tomato']
# Dictionary to store the number of correct predictions for each class
correct = dict()
for name in product_names:
    count = 0
    for img_name in os.listdir('D:\\ImageNet\\dataset\\training_set\\second_attempt\\' + name + '\\testing'):
        test_image = image.load_img(img_name, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        index = np.where(result==1)
        if index[1] != []:
            count += 1
    correct[name] = count
print(correct)