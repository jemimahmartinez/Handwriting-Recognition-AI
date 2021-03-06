# CS302-Python-2020-Group7

***NOTE:***
Our original code for testing the HASYv2 dataset is within the master branch.
To get to our code for our MNIST dataset, please refer to the 'mnist' branch

<ins>Purpose of the System:</ins>

This project aims to increase our understanding of machine learning and supervised learning by developing an AI System for image classification. Our team goal is to develop an AI system which focuses on handwriting recognition. 
This AI system we are developing could potentially be used as an application to convert handwritten phone numbers to a digital format.  

<ins>Database to use:</ins>

The database chosen for our system is the HASYv2 dataset which is a handwritten/symbol database. This database is an existing one that has 32px x 32px centred images of 369 symbol classes. It has a total of 168,233 images of handwritten symbols such as handwritten alphanumeric chars - numbers, letters, mathematical and scientific symbols.However, due to unforeseen circumstances and limitations regarding hardware, we have changed our database to MNIST. This database has 28px x 28x centred images. It has a total of 60,000 training images and 10,000 testing images consiting of the handwritten arabic numerals, 0  to 9. We decided to choose this database to work with as it does not take up as much memory as the HASYv2 dataset.

<ins>Testing plan:</ins>

We have selected 4 different CNN models (LeNet5, AlexNet, VGG16 and ResNeXt) to test our images with. We will be using different sets of images for the AI training and AI testing. We have ensured that our data and our network will all be on the same device as to avoid unwanted errors. The testing environment of our network will be on a device with an Intel Core i7-65000U Processor and with CUDA 10.1 using GPU acceleration, an NVIDIA driver (GeForce 920M) which will be accessed so that the libraries in CUDA will allocate tasks to our GPU accordingly. With the CPU and GPU of our device, the rate at which data is being transferred through the system will be high, allowing for a greater amount of calculations to run concurrently and thus allowing faster training. 

<ins>Results and Evaluation:</ins>

From the four models that we implemented, the one we have chosen to pursue is the LeNet-5 model. 
We conducted tests and checks for accuracy and loss as well as a confusion matrix to retrieve values for Precision, Recall, and F1-score.

Accuracy: 74%  

![ ](https://github.com/UOA-CS302-2020/CS302-Python-2020-Group7/blob/master/results/accuracy_lenet5.png)


Loss: 0.79

![ ](https://github.com/UOA-CS302-2020/CS302-Python-2020-Group7/blob/master/results/loss_LeNet5.png)


Precision, Recall, F1 Score:

![ ](https://github.com/UOA-CS302-2020/CS302-Python-2020-Group7/blob/master/results/ppf1_lenet5.png)
