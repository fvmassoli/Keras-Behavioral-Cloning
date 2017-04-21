from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pydot
from utils import read_file, prepare_dataset
from keras.utils import plot_model
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

base_path = 'simulator_outputs/training_20_04_2017'
lines = read_file(base_path=base_path)
features, labels = prepare_dataset(base_path=base_path, lines=lines)

NB_EPOCHS=15
BATC_SIZE=128
LEARNING_RATE=0.0001
VALIDATION_SPLIT=0.2
REGULARIZATION_FACTOR=0.001
DROPOUT_PROB=0.3

fc1_output_size = 1164
fc2_output_size = 100
fc3_output_size = 50
fc4_output_size = 10
fc5_output_size = 1

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,24),(60,60))))
model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Convolution2D(48, (5, 5), strides=(2,2), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Convolution2D(64, (3, 3), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Convolution2D(64, (3, 3), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add((Flatten()))
model.add(Dense(1164, activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(100, activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(50, activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(10, activation='elu', kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',#Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=1.e-4,
                                        patience=1,
                                        mode='min')
history_object = model.fit(features,
                           labels,
                           epochs=NB_EPOCHS,
                           batch_size=BATC_SIZE,
                           validation_split=VALIDATION_SPLIT,
                           shuffle=True,
                           verbose=1,
                           callbacks=[early_stopping_callback])

model.summary()
model.save('model2.h5')
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model_loss_hisotry.png')
