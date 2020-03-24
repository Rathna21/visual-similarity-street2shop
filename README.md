## VISUAL SIMILARITY USING DEEP METRIC LEARNING FOR STREET2SHOP DATASET ## 


This repository contains the training and inferencing code for Visual Similarity and matching for Street2Shop dataset using deep metric learning and triplet loss.

The goal is to match a real-world example of a garment item to the same item in an online shop. This is an extremely challenging task due to visual differences between street photos, pictures of people wearing clothing in everyday uncontrolled settings, and online shop photos pictured on people, mannequins, or in isolation, captured by professionals. 

DATASET: Street2shop dataset contains exact street2shop pairs and the retrieval sets for 11 clothing categories: 
    - bags
    - belts
    - dresses
    - eyewear
    - footwear
    - hats
    - leggings
    - outerwear
    - pants
    - skirts
    - tops
    
Here we focus only on one category i.e., 'dresses' amoung the above different categories. Further information about the dataset is found here http://www.tamaraberg.com/street2shop/. 


### Repository overview ### 

The files necessary for training are in train folder and the inferencing notebook is in the test folder. 
1. train/
    - model.py -- script for model definition
    - metrics.py -- script for custom loss function and accuracy in Keras 
    - utils.py -- script having all utility functions
    - generator.py -- script for generating triplets for training 
    - triplet_training.py -- script for training the deep learning model
    
2. test/
    - Inferencing.ipynb -- jupyter notebook for inferencing the trained model on the test images and evaluating the performance. 
    
### Training ###

Run the file train/triplet_training.py for training the model. 

Command: python train/triplet_training.py



