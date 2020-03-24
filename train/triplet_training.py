# Import necessary packages 

import pickle
import os
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import (Dense, Dropout, Input, Lambda, GlobalAveragePooling2D, concatenate, BatchNormalization, Activation)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard )
import numpy as np
import math
import random
from PIL import Image
import pandas as pd
from utils import get_layers_output_by_name, get_callbacks, convert_data, load_old_model, get_image_generator
from model import base_network_vgg, triplet_network
from metrics import lossless_triplet_loss, triplet_accuracy
from generator import  s2s_data_generator

# Config GPU 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# load all necessary files from disk
with open('./train_test_final_triplets.pkl', 'rb') as f:
    final_triplets = pickle.load(f)
    
with open('./bbox_mappings_streetImg.pkl', 'rb') as f:
    bbox_mappings = pickle.load(f)

with open('./train_test_s2s_full.pkl', 'rb') as f:
    duplets = pickle.load(f)

with open('./all_catalog_df.pkl', 'rb') as f:
    catalog_images = pickle.load(f)
    

img_size = (300, 300, 3)
batch_size = 32
n_epochs = 100
N = 128
k = 10

# optimizer
opt = Adam(lr=0.0002)

datagen = get_image_generator()

# define triplet network model
base_model = base_network_vgg(input_shape=img_size)
triplet_model = triplet_network(base_model=base_model, input_shape=img_size)
print("Base model:")
base_model.summary()
print("\nTriplet model:")
triplet_model.summary()

# callbacks
callbacks = get_callbacks(
    "./models/dresses-s2s-iteration1-ckpt.h5",
    initial_learning_rate=0.0002,
    learning_rate_drop=0.5,
    learning_rate_epochs=50,
    learning_rate_patience=50,
    early_stopping_patience=50)

triplet_model.compile(optimizer=opt, loss=lossless_triplet_loss)

# data generator
training_generator = s2s_generator(
    duplets,
    catalog_images,
    batch_size=batch_size)


# fit the model
history = triplet_model.fit_generator(
    generator=training_generator,
    steps_per_epoch= len(duplets) / batch_size
    epochs=n_epochs,
    callbacks=callbacks)

triplet_model.save("./models/dresses-s2s-iteration1.h5")
pickle.dump(history.history, open('./models/dresses-s2s-iteration1-history.pkl', 'wb'))