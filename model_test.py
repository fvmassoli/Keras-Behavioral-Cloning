from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from utils import read_file, preprare_dataset

base_path = 'simulator_outputs/track_1/'
lines = read_file(base_path)
features, labels = preprare_dataset(base_path, lines)

NB_EPOCHS=3
BATC_SIZE=128
LEARNING_RATE=0.001
VALIDATION_SPLIT=0.2
REGULARIZATION_FACTOR=0.001
DROPOUT_PROB=0.3

activation_function = 'relu'

fc1_output_size = 1164
fc2_output_size = 100
fc3_output_size = 50
fc4_output_size = 10
fc5_output_size = 1

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
#model.add(Cropping2D(cropping=((70,24),(0,0))))
model.add(Convolution2D(24, (5, 5), strides=(2,2), activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Convolution2D(48, (5, 5), strides=(2,2), activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Convolution2D(64, (3, 3), activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Convolution2D(64, (3, 3), activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add((Flatten()))
model.add(Dense(1164, activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(100, activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(50, activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(10, activation=activation_function, kernel_regularizer=l2(REGULARIZATION_FACTOR)))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(1))
model.compile(loss='mse',
               optimizer=Adam(lr=LEARNING_RATE),
               metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1, verbose=0, mode='min')
history_object = model.fit(
                    features, labels,
                    epochs=NB_EPOCHS,
                    verbose=1,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=[early_stopping_callback])
# history_object = model.fit(
#                     features, labels,
#                     batch_size=BATC_SIZE,
#                     nb_epoch=NB_EPOCHS,
#                     verbose=1,
#                     shuffle=True,
#                     validation_split=0.2,
#                     callbacks=[early_stopping_callback])

#score = model.evaluate(X_test, y_test, verbose=0)

# print a summary representation of your model
# model.summary()

#print('Test score:', score[0])
#print('Test accuracy:', score[1])

# save the model
model.save('model.h5')

#print('Model history: ' + str(history_object.history))
#print('Model history keys: ' + str(history_object.history.keys()))

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model_hisotry.png')