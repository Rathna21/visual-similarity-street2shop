
import os
import sys
import shutil
from PIL import Image
import numpy as np
from keras.callbacks import (ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard )
from functools import partial
import math
# from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from metrics import lossless_triplet_loss, triplet_accuracy


def merge_directories(original_dir, target_dir):
    
    """
    A function to walk through a list of directories and merge them. 
    
    Arguments: 
    original_dir -- Path to the original root directory 
    target_dir -- Path to new target directory
    
    """
    
    for root, dirs, files in os.walk(original_dir, followlinks=True):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(target_dir, file)
            shutil.copy(src_path, dest_path)


def convert_to_numpy(dir_name, img_list):
    
    """
    A function to convert a list of images to numpy array.
    
    Arguments:
    dir_name -- directory path to the list of images
    img_list -- list of all images 
    
    Returns:
    numpy_arr of all images having dimension (len(images), width, height, channels)
    
    """
    image_arr = list()
    for im in img_list:
        im_name = os.path.join(dir_name, im)
        im = np.array(Image.open(im_name))
        image_arr.append(im)
        
    return np.array(image_arr)
        
def get_layers_output_by_name(model, layer_names):
    
    """A function to return the outputs of specified layers in the model."""
    return {v: model.get_layer(v).output for v in layer_names}
    
    
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    """A function for learning rate scheduler."""
    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))


def get_image_generator():
    
    """Image data generator for training dataset performing various data transformations."""
    datagen_args = dict(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        fill_mode="nearest",
        horizontal_flip=True)
    
    return ImageDataGenerator(**datagen_args)


def get_callbacks(
    model_file,
    initial_learning_rate=0.0001,
    learning_rate_drop=0.5,
    learning_rate_epochs=None,
    learning_rate_patience=50,
    logging_file="fashion_street2shop.log",
    verbosity=1,
    early_stopping_patience=None,
):
    """
    Keras callbacks.
     
    """
    callbacks = list()
    callbacks.append(
        TensorBoard(
            log_dir="./logs",
            histogram_freq=0,
            batch_size=10,
            write_graph=False,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            embeddings_data=None,
            update_freq="batch",
        )
    )
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True, period=1))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(
            LearningRateScheduler(
                partial(
                    step_decay,
                    initial_lrate=initial_learning_rate,
                    drop=learning_rate_drop,
                    epochs_drop=learning_rate_epochs,
                )
            )
        )
    else:
        callbacks.append(
            ReduceLROnPlateau(
                factor=learning_rate_drop,
                patience=learning_rate_patience,
                verbose=verbosity,
            )
        )
    if early_stopping_patience:
        callbacks.append(
            EarlyStopping(verbose=verbosity, patience=early_stopping_patience)
        )
    return callbacks

def convert_data(q_list, p_list, n_list, dummy_list):
    
    """Create pre-processed batch arrays. Pre-process the input based the specified base model."""
    q = preprocess_input(np.asarray(q_list))
    p = preprocess_input(np.asarray(p_list))
    n = preprocess_input(np.asarray(n_list))
    dummy = np.asarray(dummy_list)
    return [q, p, n], dummy

def load_old_model(model_file):
    
    """Load pretrained model"""
    print("Loading pre-trained model")
    return load_model(model_file, custom_objects={"lossless_triplet_loss": lossless_triplet_loss})


def get_image_features(model, im, N=2048):
    
    """ Get image features for an image during inferencing"""
    image_features = np.zeros((1, N))
    im = im.astype(np.float32, copy=False)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    image_features[0, :] = model.predict([im, im, im])[0, 0, :N]
    return image_features[0]

def get_result(image_embedding, search_index, k=10):
    
    """ Get the k nearest neighbours result for the given image embedding"""
    kNN = search_index.get_nns_by_vector(image_embedding, k + 1, include_distances=True)
    return kNN[0]

