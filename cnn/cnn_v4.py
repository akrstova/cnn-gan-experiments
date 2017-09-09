from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation
from keras.layers.core import Dropout
import datetime

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu'))

classifier.add(Flatten())
classifier.add(keras.layers.core.Dropout(0.5))
classifier.add(Dense(units=256, activation='relu'))
classifier.add(keras.layers.core.Dropout(0.5))
classifier.add(Dense(units=4, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

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

start = datetime.datetime.now()
classifier.fit_generator(training_set, steps_per_epoch=250, epochs=25, validation_data=test_set, validation_steps=65)
end = datetime.datetime.now()
print("Finished in {} minutes" % (end.minute - start.minute))

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# Encapsulate building of classifier into a function
def build_classifier(optimizer):
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    # Add the max pooling layer to reduce the feature maps
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # classifier.add(Dropout(0.25))

    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # classifier.add(Dropout(0.25))

    # Add flattening
    classifier.add(Flatten())

    # Add fully connected layer
    classifier.add(Dense(units=128, activation='relu'))

    # Add output layer
    classifier.add(Dense(units=4, activation='sigmoid'))

    # Compile
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Grid search to find the optimal parameters
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(training_set, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_