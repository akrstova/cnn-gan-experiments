# Improve the previous architectures by adding a dropout for normalization

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation
from keras.layers.core import Dropout
import datetime

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(keras.layers.core.Dropout(0.25))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(keras.layers.core.Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 4, activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set/second',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'test_set/second',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

start = datetime.datetime.now()
classifier.fit_generator(training_set, steps_per_epoch=250, epochs=25, validation_data=test_set, validation_steps=65)
end = datetime.datetime.now()
print("Finished in {} minutes" % (end.minute - start.minute))