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

In all the layers a RELU activation function has been used and L2 regularization has been applied in order to penalize larger weights and break down possible degeneracy problems.

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

####1. Solution Design Approach

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              1342092   
_________________________________________________________________
dropout_1 (Dropout)          (None, 1164)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 1,595,511
Trainable params: 1,595,511
Non-trainable params: 0


The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text](https://github.com/fvmassoli/fem-CarND-LaneLines-P1/blob/master/outputs/gray_scale.png "Grayed image")

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
