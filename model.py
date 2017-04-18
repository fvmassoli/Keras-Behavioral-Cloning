import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
#from keras.initializers import TruncatedNormal, Zeros
import matplotlib.pyplot as plt
from utils import read_file, preprare_dataset, training_data_generator, validation_data_generator

base_path = 'simulator_outputs/track_1/'
lines = read_file(base_path)
features, labels, input_shape = preprare_dataset(base_path, lines)
print('Input shape: ' + str(input_shape))

NB_EPOCHS = 1
BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 64
DROPOUT_PROB = 0.5
REGULARIZATION_FACTOR = 0.02
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.3

nb_kernels_1 = 3
nb_kernels_2 = 24
nb_kernels_3 = 36
nb_kernels_4 = 48
nb_kernels_5 = 64

kernel_size_1 = 5
kernel_size_2 = 3

stride_1=(2, 2)
stride_2=(1, 1)

weights_initializer = None#TruncatedNormal(mean=0.0, stddev=1.e-2, seed=None)

biases_initializer = None#Zeros

activation_function = Activation('elu')

fc1_output_size = 1164
fc2_output_size = 100
fc3_output_size = 50
fc4_output_size = 10

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(nb_kernels_1, 5, 5, activation='elu'))
# model.add(Convolution2D(nb_kernels_2, 5, 5, activation='elu'))
# model.add(Convolution2D(nb_kernels_3, 5, 5, activation='elu'))
# model.add(Convolution2D(nb_kernels_4, 3, 3, activation='elu'))
# model.add(Convolution2D(nb_kernels_5, 3, 3, activation='elu'))
model.add((Flatten()))
#model.add(Dense(fc1_output_size))
#model.add(Dropout(DROPOUT_PROB))
# model.add(Dense(fc2_output_size))
# model.add(Dropout(DROPOUT_PROB))
# model.add(Dense(fc3_output_size))
# model.add(Dropout(DROPOUT_PROB))
# model.add(Dense(fc4_output_size))
# model.add(Dropout(DROPOUT_PROB))
# model.add(Dense(fc5_output_size))
# model.add(Flatten())
# model.add(Dense(1))
model.add(Dense(1))

model.compile(loss='mse',
               optimizer='adam',
               metrics=['accuracy'])

history = model.fit(features,
                    labels,
                    0.2,
                    shuffle=True,
                    nb_epoch=NB_EPOCHS)


#
# print('Start training...')
#
# training_generator = training_data_generator(features, labels)
# validation_enerator = validation_data_generator(features, labels)

#checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1.e-4, verbose=1, mode='auto')
#
# model.compile(loss='mse',
#                optimizer=Adam(lr=LEARNING_RATE),
#                metrics=['accuracy'])

# history = model.fit_generator(training_generator,
#                                steps_per_epoch=batch_size,
#                                epochs=nb_epochs,
#                                validation_data=validation_enerator,
#                                validation_steps=validation_batch_size,
#                                callbacks=[checkpoint, early_stopping])
#
# history = model.fit(features,
#                     labels,
#                     validation_split=VALIDATION_SPLIT,
#                     shuffle=True,
#                     nb_epoch=NB_EPOCHS,
#                     verbose=1)
#                    callbacks=EarlyStopping(monitor='val_loss',
#                                            min_delta=1.e-4,
#                                            patience=2,
#                                            verbose=1,
#                                            mode='auto'))
#
# model.save('model.h5')
#
# print('Neural Net ready!!!')







#print(history.history)

# list all data in history
#print(history.history.keys())

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()