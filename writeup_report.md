# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 containing a video that show the performances of the neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the implementation of the CNN using keras. There is also an utils.py file which contains two methods for read data from the .csv file and load the images. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried several architectures and since I got the best performances with the NVIDIA one I used that model in my project (lines 32-48).

From an overall point of view, the CNN is made by a Lambda layer and a Cropping layer that normalizes and crops the images, respectively.
There are then 5 convolutional layers: the first three with a 5x5 kernel and 2x2 stride and the last two with a 3x3 kernel and no stride.
The depth of the 5 layers are 24, 36, 48, 64 and 64, respectively. 

4 Dense layers are built on top of the convolutional ones with sizes of 1164, 100, 50,  10. After them there is the output node.

In total there are 1.595.511 trainable parameters in the net.

In all the layers a RELU activation function has been used and L2 regularization has been applied in order to penalize larger weights and break down possible degeneracy problems.

I tested also the ELU activation function without finding any concrete advantage in using it.

#### 2. Attempts to reduce overfitting in the model

In order to avoid overfitting I added Dropout() layers with prob=0.3 to the fully connected layers. Moreover, I set the EarlyStopping callback (line 55-58) in order to stop the training if needed.

I tested the use of generators in order to augment the data I recorder by driving along the track. By the way, using generators really slowed down the training. Since I have enough resources, I then decided to increase the acquired data and to produce new images and then train the network without the use of generators.

#### 3. Model parameter tuning

In order to evaluate the loss of the CNN I used mse function coupled with Adam() optimizer (line 51-53). 

#### 4. Appropriate training data

At the beginning I tried to acquire data by driving as close as possible to the center of the street. By the way, such a strategy required a lot of small steering adjustments. Train the model with that data resulted in a car that continuously shook from one side to other of the street. A better approach has been to try to steer much less and let the car move straight as much as possible even though it was displacing from the exact center of the road. Thar strategy worked much better in order to train the data. Using such a concept I acquired two laps driving clockwise and two laps counterclockwise. I also recorded few times a "hard recovery" from the street side.

During the training I used the center, left and right camera images taking into account the appropriate steering angle correction (utils.py file, lines 57-73).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a very simple NN made of two convolutional layer and two fully connected layers. I had some good results but after few turns the car usually crashed and went out of the street. I tested few more models but at the end the NVIDIA model gave the best performances. The model has been implemented in the model.py file (line 31-49).

I prepared the dataset by recording two laps driven in opposite direction plus other "hard recoveries" and "bridge crossings" recordings. I then split the data among training and validation set (20%). 

In order to augment the data I tried to use python generators but then the model training was very slow!! For that reason I decided to train my model using all the resources I had and so I produced new images by flipping, adding random shadows and by changing the brightness of the original images. Having the images already produced it usually takes between 10 to 20 minutes to train the model.

I tested different values for training parameters such as validation percentage, batch size, epochs, learning rate, dropout probability, etc. I finally chose the values that are implemented in the code. 

I tested the RELU() and ELU() activation functions and since I didn't obtain the expected improvement by using the ELU() I decided to use the RELU() function. 

I found a big improvement by cropping vertically the image, i.e. removing several pixels columns. In particular it improved upon some weird behavior such as turning continuously left and right of the car. In order to crop the input image I used a Cropping2D layer and I finally obtained an input shape of 66x200.

In order to avoid overfitting I used dropout layers with prob=0.3 applied to the fully connected layers and the early stopping callback in order to stop the training as soon as no improvement happened among two consecutive epochs.


 
#### 2. Final Model Architecture

As explained in a previous section the model consists of one Lamba layer + 1 cropping layer + 5 convolutional layers + 4 Dense layers.

Here is a visualization of the architecture

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/model.png "Model")

Here is a gif that shows the car driven by the CNN:

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/behavioral_cloning.gif "")


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/center_driving.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in case it gets too close the street border. These images show what a recovery looks like starting from the right side of the street:

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/center_2017_04_20_17_18_44_574.jpg)
![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/center_2017_04_20_17_18_46_017.jpg)
![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/center_2017_04_20_17_18_46_567.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the dataset I flipped, randomly added shadows and changed the brightness of the images. For example, here is an image that has been modified:

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/img/original.jpg "Original image")
![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/img/flipped.jpg "Flipped image")
![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/img/shadowed.jpg "Shadowed image")
![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/img/augmented_brightness.jpg "Augmented brightness image")
 
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped to determine if the model was over or under fitting. The ideal number of epochs was 12 since after such a number of epochs, usually the early termination callback was called. I used an adam optimizer so that manually training the learning rate wasn't necessary.
