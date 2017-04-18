from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import numpy as np

def preprocess_data(data):
    img = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    img = img/255.0 - 0.5
    return img

datagen = RegressionImageDataGenerator(
    rotation_range=40,
    horizontal_flip_value_transform=lambda val: -val,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

img = load_img('simulator_outputs/track_1/IMG/center_2017_04_13_22_43_12_507.jpg')



x = img_to_array(img)
x = preprocess_data(x)
print(x.shape)
x = x.reshape((1,) + x.shape + (1,))
print(x.shape)

i = 0
for b in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='test', save_format='jpeg'):
    i += 1
    if i > 5:
        break