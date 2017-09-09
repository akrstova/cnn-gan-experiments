# Experiments with Convolutional and Generative Adversarial Neural Networks in Keras

Code for the semester project in Intelligent Information Systems documenting the experimental results described in the research paper *Image Recognition using Convolutional Neural Networks in the context of mobile applications*. 

## Authors

**Alisa Krstova**, 141501

**Alek Petreski**, 141507

## Background and previous work

Our goal is to integrate a deep learning model which will be able to distinguish between different kinds of fruit, vegetables and other food products into our Android application for finding recipes. We believe that having a system which will recognize food products via pictures taken from the user's camera would be a good solution to the UI & UX problem of entering the ingredients to be used for cooking. 
Our first implementation of this deep learning model involves classification between 10 different products with the use of CNN. The model is trained on images from ImageNet.

## Structure

The project contains documented code for 6 different architectures of Convolutional Neural Networks written in Keras. The 6th version of our CNN code gives the best results on the training and validation data.
In the directory **gan** you can find an implementation of a Generative Adversarial Neural Network model, whose goal is to generate new images based on ImageNet originals and thus augment the dataset for the aforementioned CNN classification model.


