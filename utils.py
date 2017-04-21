import csv
import cv2
import numpy as np
import matplotlib.image as mpimg


def read_file(base_path):
    '''
    Read the csv file containing info about dataset
    :param base_path: base path to source images
    :return: list of images path
    '''
    lines = []
    file_path = base_path + '/driving_log.csv'
    with open(file_path) as csvfile:

        reader = csv.reader(csvfile)
        i = 0
        print('Start reading .csv file')
        for line in reader:
            lines.append(line)
            if i % 500 == 0:
                print('At line: ' + str(i))
            i+=1
    return lines


def prepare_dataset(base_path, lines):
    '''
    Reads images and the respective value of the steering angle
    :param base_path: base path to source file
    :param lines: list of paths to source images
    :return: features, labels and shape of the input images
    '''
    images = []
    steering_measurements = []
    i=0
    for line in lines:

        if i % 500 == 0:
            print('Reading line: ' + str(i))
        i += 1

        center_camera_image = line[0]
        left_camera_image = line[1]
        right_camera_image = line[2]

        filename_c = center_camera_image.split('/')[-1]
        filename_l = left_camera_image.split('/')[-1]
        filename_r = right_camera_image.split('/')[-1]

        path_c = base_path + '/IMG/' + filename_c
        path_l = base_path + '/IMG/' + filename_l
        path_r = base_path + '/IMG/' + filename_r


        image_c = mpimg.imread(path_c)
        image_l = mpimg.imread(path_l)
        image_r = mpimg.imread(path_r)

        images.append(image_c)
        images.append(image_l)
        images.append(image_r)

        steering_center = float(line[3])
        steering_offset = 0.25

        steering_left = steering_center + steering_offset
        steering_right = steering_center - steering_offset

        steering_measurements.append(steering_center)
        steering_measurements.append(steering_left)
        steering_measurements.append(steering_right)

    '''
    Convert to numpy since it is the format keras expects
    '''
    features = np.array(images)
    labels = np.array(steering_measurements)

    return features, labels
