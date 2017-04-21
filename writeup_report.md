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
* run_1.mp4 containing a video that show the performances of the nueral network 
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

I tried several achitectures and since I got the best performances with the NVIDIA one I used that model in my project (lines 32-48).

From an overall point of view, the CNN is made by a Lambda layer and a Cropping layer that normalize and crop the images, respectively.
There are then 5 convolutional layers: the firs three with a 5x5 kernel and 2x2 stride and the last two with a 3x3 kernel and no stride.
The depth of the 5 layers are 24, 36, 48, 64 and 64, respectively. 

4 Dense layers are built on top of the convolutional ones with sizes of 1164, 100, 50,  10. After them there is the output node.

In total there are 1.595.511 trainable parameters in the net.

In all the layers a RELU activation function has been used and L2 regularization has been applied in order to penalize larger weights and break down possible degeneracy problems.

I tested also the ELU activation function without finding any concrete advantage in using it.

#### 2. Attempts to reduce overfitting in the model

In order to aviud overfitting I applied Dropout with prob=0.3 to the fullly connected layers.

I tested the use of generators in order to augment the data I recorder by driving along the track. By the way, using generators really slowed down the training. Since I then decided to increase the acquried data and train the network withou the use of geenrators.

Moreover, I set the EarlyStopping callback (line 54-57) in order to stop the training if needed.

#### 3. Model parameter tuning

In order to evaluate the loss of the CNN I used mse function coupled with Adam() optimizer (line50-53). 

#### 4. Appropriate training data

In order to train the model I acquire data by driving two laps clcokwise and two laps counterclockwise trying to stay as much as possible to the center of the road. I also recored few times a "hard recovery" from the street side.

During the training I used the center, left and right camera images taking into account the appropriate steering angle correction (utils.py file, lines 57-73).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a very simple NN made of two convolutional layer and two fully connected layers. I had some good results but after few turns the car usually crashed and went out of the street. I did another couple of example ending with the NVIDIA model that gave me the best performances. 

I prepared the dataset by recording two laps driven in opposite direction plus other "hard recoveries" and "bridge crossings". I then split the data ni training and validation set (20%). 

In order to augment the data I tried to use python generators but then the model training was very slow!! For that reason I decided to train my model using all the resources I had and so I produced new images by flipping and by changing the brightness of the original images. Having the images already produced it usally takes me 5 to 10 minutes to train my model.

I tested different values for validation percentage, batch suze, epochs, learning rate, etc. I finally choosed the values that are implemented in the code. 

I tested the RELU() and ELU() activation function and since I didn't obtain the expected improvement by using the ELU() I decided to use the RELU() function. 

I also found a big improvement by cropping vertically the image, i.e. removing several pixels columns. In particular it improved upon some wierd behaviour such as turning continuously left and right of the car.

Using a cropping2D layer I finally obtained an input shape of 66x200.

In order to avoid overfitting I used dropout layers with prob=0.3 applied to the fully connected layers and I use the early stopping callback in order to stop the training as soon as no improvement happened among two consecutive epochs.


 
#### 2. Final Model Architecture

As explained in a previous section the model consists of one Lamba layer + 1 cropping layer + 5 convolutional layer + 4 Dense layer.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/model.png "Model")

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/center_driving.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in case it gets too close the street border. These images show what a recovery looks like starting from the righ side of the street:

![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/center_2017_04_20_17_18_44_574.jpg)
![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/center_2017_04_20_17_18_46_017.jpg)
![alt text](https://github.com/fvmassoli/fvmassoli-CarND-Behavioral-Cloning-P3/blob/master/examples/center_2017_04_20_17_18_46_567.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped the images in order to augment the data and to avoid overfitting. For example, here is an image that has then been flipped:

![alt text]("Original image")
![alt text]("Flipped image")

 
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 12 since afetr such a number of epochs, usually the early termination callback was called. I used an adam optimizer so that manually training the learning rate wasn't necessary.
