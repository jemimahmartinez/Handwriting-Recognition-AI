"# CS302-Python-2020-Group7" 


Purpose of the System:

This project aims to increase our understanding of machine learning and supervised learning by developing an AI System for image classification. Our team goal is to develop an AI system which focuses on handwriting recognition. Currently, many AI systems and applications can detect the differences between the numbers 0 to 9 or the differences between the letters of the English language. We aim to build upon those AI systems by training our system to detect and classify numbers, letters (upper and lower case) and mathematical symbols. 
This AI system we are developing could potentially be used as an application to convert handwritten documentation to a digital format. It would be particularly useful for students to convert any handwritten documents such as notes or assignments to a digital format. 


Database to use:

The database chosen for our system is the HASYv2 dataset which is a handwritten/symbol database. This database is an existing one that has 32px x 32px centred images of 369 symbol classes. It has a total of 168,233 images of handwritten symbols such as handwritten alphanumeric chars - numbers, letters, mathematical and scientific symbols. We decided to choose this database to work with because most databases out there do not have a wide range of distinct characters that would suit our purpose. 


Testing plan:

We have selected 4 different CNN models (LeNet5, AlexNet, VGG16 and ResNeXt) to test our images with. We will be using different sets of images for the AI training and AI testing and have chosen a database ratio of 9/1, respectively. We have ensured that our data and our network will all be on the same device as to avoid unwanted errors. The testing environment of our network will be on a device with an Intel Core i7-65000U Processor and with CUDA 10.1 using GPU acceleration, an NVIDIA driver (GeForce 920M) which will be accessed so that the libraries in CUDA will allocate tasks to our GPU accordingly. With the CPU and GPU of our device, the rate at which data is being transferred through the system will be high, allowing for a greater amount of calculations to run concurrently and thus allowing faster training. 

Results and Evaluation:



