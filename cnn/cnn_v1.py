# Very first experiment with CNN in Keras
# Distinguish between tomatoes and apples
# TRAINING SET:
# Tomatoes 495
# Apples 535
# TEST SET:
# Tomatoes 212
# Apples 228
# Max accuracy obtained: 0.9975
# Max validation accuracy: 0.8447

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier = Sequential()

# Add the conv layer
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Add the max pooling layer to reduce the feature maps
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add flattening
classifier.add(Flatten())

# Add fully connected layer
classifier.add(Dense(units = 128, activation = 'relu'))

# Add output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Image preprocessing (data augmentation)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set, steps_per_epoch=250, epochs=25, validation_data=test_set, validation_steps=65)