# Brand new architecture with multiple hidden convolutional layers
# Try LeakyReLU as activation function
# More extensive data augmentation

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation
from keras.layers.core import Dropout
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
import datetime

dropout = 0.2
classifier = Sequential()

classifier.add(Conv2D(64, (3, 3), input_shape=(256, 256, 3)))
classifier.add(BatchNormalization())
classifier.add(LeakyReLU(alpha=0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(dropout))
classifier.add(Conv2D(64, (3, 3)))
classifier.add(BatchNormalization())
classifier.add(LeakyReLU(alpha=0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(dropout))
classifier.add(Conv2D(64, (3, 3)))
classifier.add(BatchNormalization())
classifier.add(LeakyReLU(alpha=0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(dropout))
classifier.add(Conv2D(64, (3, 3)))
classifier.add(BatchNormalization())
classifier.add(LeakyReLU(alpha=0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(dropout))

classifier.add(Flatten())
classifier.add(Dropout(0.2))
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
tensorboard_cb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)


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
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'test_set/second_attempt',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

start = datetime.datetime.now()
classifier.fit_generator(training_set, steps_per_epoch=250, epochs=25, validation_data=test_set, validation_steps=65)
end = datetime.datetime.now()
print("Finished in %s minutes" % (end.minute - start.minute))