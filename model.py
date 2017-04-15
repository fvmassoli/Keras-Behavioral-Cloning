import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.initializers import TruncatedNormal, Zeros
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import matplotlib.pyplot as plt


def read_file(base_path):
    lines = []
    with open(base_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def preprare_dataset(base_path, lines):
    images = []
    steering_measurements = []
    for line in lines:
        center_camera_image = line[0]
        left_camera_image = line[1]
        right_camera_image = line[2]

        filename_c = center_camera_image.split('/')[-1]
        filename_l = left_camera_image.split('/')[-1]
        filename_r = right_camera_image.split('/')[-1]

        path_c = base_path + 'IMG/' + filename_c
        path_l = base_path + 'IMG/' + filename_l
        path_r = base_path + 'IMG/' + filename_r

        image_c = cv2.imread(path_c)
        image_l = cv2.imread(path_l)
        image_r = cv2.imread(path_r)

        images.append(image_c)
        images.append(image_l)
        images.append(image_r)

        steering_center = float(line[3])
        steering_offset = 0.05

        steering_left = steering_center - steering_offset
        steering_right = steering_center + steering_offset

        steering_measurements.append(steering_center)
        steering_measurements.append(steering_left)
        steering_measurements.append(steering_right)

    X_train = np.array(images)
    input_shape = X_train.shape[1:]

    y_train = np.array(steering_measurements)

    return X_train, y_train, input_shape


base_path = 'simulator_outputs/track_1/'
lines = read_file(base_path)
X_train, y_train, input_shape = preprare_dataset(base_path, lines)

print(input_shape, 'input shape')

nb_kernels_1 = 24
nb_kernels_2 = 36
nb_kernels_3 = 48
nb_kernels_4 = 64

kernel_size_1 = 5
kernel_size_2 = 3

stride=(2, 2)

weights_initializer = TruncatedNormal(mean=0.0, stddev=1.e-2, seed=None)

biases_initializer = Zeros

activation_function = Activation('elu')

dropout_prob = 0.5

fc1_output_size = 1164
fc2_output_size = 100
fc3_output_size = 50
fc4_output_size = 10

regularization_factor = 0.02

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5), input_shape=input_shape)
# crop the 50 most top pixels rows
model.add(Cropping2D((50, 0), (0, 0)))

model.add(Convolution2D(nb_kernels_1,
                        kernel_size_1,
                        strides=stride,
                        padding='valid',
                        kernel_initializer=weights_initializer,
                        bias_initializer=biases_initializer,
                        activation=activation_function,
                        W_regularizer=regularizers.l2(regularization_factor)))

model.add(Convolution2D(nb_kernels_2,
                        kernel_size_1,
                        strides=stride,
                        padding='valid',
                        kernel_initializer=weights_initializer,
                        bias_initializer=biases_initializer,
                        activation=activation_function,
                        W_regularizer=regularizers.l2(regularization_factor)))

model.add(Convolution2D(nb_kernels_3,
                        kernel_size_2,
                        strides=stride,
                        padding='valid',
                        kernel_initializer=weights_initializer,
                        bias_initializer=biases_initializer,
                        activation=activation_function,
                        W_regularizer=regularizers.l2(regularization_factor)))

model.add(Convolution2D(nb_kernels_4,
                        kernel_size_2,
                        strides=stride,
                        padding='valid',
                        kernel_initializer=weights_initializer,
                        bias_initializer=biases_initializer,
                        activation=activation_function,
                        W_regularizer=regularizers.l2(regularization_factor)))

model.add(Flatten())

model.add(Dense(fc1_output_size,
                kernel_initializer=weights_initializer,
                bias_initializer=biases_initializer,
                activation=activation_function))
model.add(Dropout(dropout_prob))

model.add(Dense(fc2_output_size,
                kernel_initializer=weights_initializer,
                bias_initializer=biases_initializer,
                activation=activation_function))
model.add(Dropout(dropout_prob))

model.add(Dense(fc3_output_size,
                kernel_initializer=weights_initializer,
                bias_initializer=biases_initializer,
                activation=activation_function))
model.add(Dropout(dropout_prob))

model.add(Dense(fc4_output_size,
                kernel_initializer=weights_initializer,
                bias_initializer=biases_initializer,
                activation=activation_function))
model.add(Dropout(dropout_prob))

model.add(Dense(1))





datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
)











print('Start training...')

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    validation_split=0.2,
                    shuffle=True,
                    nb_epoch=2,
                    verbose=1)

print('Neural Net ready!!!')
model.save('model.h5')

print(history.history)

# list all data in history
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()