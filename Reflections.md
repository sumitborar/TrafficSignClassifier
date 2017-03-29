#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the image data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


###1.Data Set Summary & Exploration
Data consists of 32x32 size images of Traffic signs with labels. Each image is represented in RGB form.
There are 43 different types of Traffic signs included in the data set.
Data provided is split into 3 parts.
1. Number of training examples = 178493
2. Number of validation examples = 12630
3. Number of testing examples = 12630
4. Image data shape = (32, 32, 3)
5. Number of classes = 43
The code for this step is contained in the third code cell of the IPython notebook.

###2.Visualization of the dataset and identify where the code is in your code file

The bar chart below is histogram of number of training examples for each class in the training set.

<img src="doc_images/training_samples_per_class.png" width="480" alt="Histogram of training samples per class" />
![alt text][./doc_images/training_samples_per_class.png]

The code for this step is contained in the fourth code cell of the IPython notebook.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Image is converted from RGB space to YUV space. Several vision research papers typically use this as part of the image pipeline. It encodes a color image taking human perception into account and allows bandwidth for chrominance components. This results in masking of transmission errors.

'Y' channel represents  luminosity/brightness in the image. I normalize this channel using CLHAE histogram normalization to take away effect of brightness variations.

As next step, image values are normalized and converted from 0-255 range to 0-1 range. This normalization of values helps during training optimization by converging to minima faster.

The code for this step is contained in the fifth code cell of the IPython notebook.

Example of traffic sign before and after processing.

<img src="/doc_images/image_color_normalization.png" width="480" alt="Before and After Color Normalization" />
![alt text][./doc_images/image_color_normalization.png]



####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

As we can see from sample distribution across classes, some classes have very few training samples. Also there is wide variation in samples for several classes.

I augment training and validation set by applying following process.

1. For each image in the training set, we generate 10 images.
2. Each generated image is distorted randomly based on factors like rotation, translation and shear. Affine transformations like these maintains collinearity of points and ratios of distances. This process adds variations in trainng and makes the model more robust.
3. From this augmented image set, 25% of randomly selected images are added to the validation set and rest are added to training set. After this step validation set has 91408 images.
4. In training set, a hard cutoff of 6000 samples per image is kept. We want to ensure that ratio of samples in popular classes to rare classes in maintained and at the same time not bias the classifier too much towards popular classes. This cut-off was reached by estimation and can be improved with experimentation
5. New distribution of training set samples per class is shown below

<img src="/doc_images/training_samples_per_classs.png" width="480" alt="Histogram of training samples per class" />
![alt text][./doc_images/training_samples_per_class2.png]


The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

Here is an example of an original image and an augmented image:
Original Image
<img src="/doc_images/transformed_image_seed.png" width="480" alt="Histogram of training samples per class" />
![alt text][./doc_images/transformed_image_seed.png]

Example of transformed images
img src="/doc_images/transformed_images.png" width="480" alt="Histogram of training samples per class" />
![alt text][./doc_images/transformed_images.png]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


I tested my model with two broad types of architectures.

1. Inception Model
2. Lenet Model

I was achieving almost similar accuracy numbers with both the models. Though it took really long time to train inception model. As a result my final report is generated on Lenet model. LeNet Model Architecture


<img src="/doc_images/LeNet_Arch.png" width="480" alt="LeNet Architecture" />
![alt text][./doc_images/LeNet_Arch.png]


The code for my final model is located in the seventh cell of the ipython notebook.


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook.

Adams optimizer is used with learning rate of 0.001 and epsilon of 0.1. I also experimented with learning rate of 0.005.
Batch size - 128 with 80 Epoch to train. Smaller batch size helped in converging faster and 80 epochs were derived after observing the training and validation accuracies graph.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I took an iterative approach towards a final solution. While the current literature shows VGG Net and Inception architecture provide good results, I was keen to experiment and move towards those solutions. I started with basic LeNet architecture and got roughly 94% accuracy. After that I modified LeNet by adding 1X1 layers to increase depth which resulted in increase in accuracy. Dropout operation was added to each of the fully connected layers to reduce overfitting.

In addition I also experimented with Inception architecture. In my experiments it took long time to train and gave similar results. I

After 80 Epochs,

Train accuracy - 98.2%
Validation accuracy - 97.4%
Test accuracy -

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:
','test_img2.jpg','test_img3.jpeg','test_img4.jpg','test_img5.jpeg','test_img6.jpg'
![Image 1 ][./test_images/test_img1.jpg] ![Image 2 ][./test_images/test_img2.jpg]  
![Image 3 ][./test_images/test_img3.jpeg] ![Image 4 ][./test_images/test_img4.jpg]
![Image 5 ][./test_images/test_img1.jpeg] ![Image 6 ][./test_images/test_img6.jpg]

I resized these images and pre-processed them before passing them through the classifier. On this set classifier got an accuracy of 66.67% and was unable to classify two images correctly.
Image 5 is taken at a distance and the sign is at an angle more then 20 degrees.
Image 3 is also not classified correctly. It is being mis-identified as a stop sign. I am unsure as to the reason behind this.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

![Predictions ][./doc_images/predictions_internet.png]

Model was correctly able to predict for 4/6 images. As explained earlier both the images had significant differences compared to images used in training and testing. The other four images were similar in nature to the testing images.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

As we can see from the above visualization, model is very sure of the predictions in case of Images 1,2 and 3. In case of image 4 (
"Speed Limit 60" sign ) model has some small probability over classes 2 & 5 which are similar speed limit signs. 
In Image 5 model is not able to find a good class for the image and the probabilities are spread over multiple classes with highest being 35%.
In case of Image 6 ( "Yield" sign ) model predicts with 80% probability the correct class. It also assigns about 18% probability to "Ahead Only" class
