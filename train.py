from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(64, (3, 3), input_shape = (64,64,3),padding = 'same',activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))

classifier.add(Convolution2D(128, (3, 3),padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))

classifier.add(Convolution2D(256, (3, 3),padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))

classifier.add(Flatten())

classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=40,
                                   fill_mode='nearest',
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size=(64, 64),
                                                 batch_size=10,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size=(64, 64),
                                            batch_size=10,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=200//10,
                         validation_data=test_set,
                         epochs=100,
                         validation_steps=90//10)
                         
classpath = "modelMask.hdf5"
classifier.save(classpath)                                                                                                                                                                                     