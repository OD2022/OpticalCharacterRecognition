# OpticalCharacterRecognition
Optical character recognition (OCR) is a technology that enables the recognition of texts and characters(alphabets, numbers, e.t.c) in an image. This project applies data engineering and data preparation technqiues in tandem with a deep convolutional network to train a model for recognizing english characters in images.
All the data used for training and testing can be found here: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format 

Main.py contains the code for creating the convolutional neural network, using Tensorflow, the end result is a model that can be shared or saved. Adjust the CNN settings for epochs as you dim fit, depending on how many times you want to train the model.
Preprocessing_pipeline.py contains code for segmenting images of text into characters, blur settings, dilation and contour settings as needed. Import your trained model gotten from executing main.py into Preprocessing.py to complete the pipeline.
