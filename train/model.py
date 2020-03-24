from utils import get_layers_output_by_name
from keras import backend as K
from keras.models import Model
from keras.layers import (
    Dense,
    Dropout,
    Input,
    Lambda,
    GlobalAveragePooling2D,
    concatenate,
    BatchNormalization,
    Activation
)
# from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

def base_network_vgg(input_shape=(50, 50, 3), weights="imagenet"):
    
    """
    Define base network as VGG for triplet neural network.
    
    Arguments:
    input_shape -- Input size of the base network, minimum to be (50, 50, 3)
    weights -- weights to be used by the base network, can take two values - 'None' or 'imagenet'
               'None' - random initialization of weights 
               'imagenet' - initialize imagenet weights 
    Returns:
    model -- returns the defined Keras model object
    
    """
    base = VGG16(
        include_top=False, weights=weights, input_shape=input_shape, pooling="avg")

    # frozen layers
    for layer in base.layers[:10]:
        layer.trainable = False

    # intermediate layers
    layer_names = ["block1_pool", "block2_pool", "block3_pool", "block4_pool"]
    intermediate_layer_outputs = get_layers_output_by_name(base, layer_names)
    convnet_output = base.output
    for layer_name, output in intermediate_layer_outputs.items():
        output = GlobalAveragePooling2D()(output)
        convnet_output = concatenate([convnet_output, output])

    # top layers
    convnet_output = Dense(2048, activation="relu")(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Dense(2048, activation="relu")(convnet_output)
    convnet_output = Lambda(lambda x: K.l2_normalize(x, axis=1))(convnet_output)

    model = Model(inputs=base.input, outputs=convnet_output, name="base_network")

    return model

def base_network_resnet(input_shape=(50, 50, 3), weights="imagenet"):
    
    """
    Define base network as ResNet for triplet neural network.
    
    Arguments:
    input_shape -- Input size of the base network, minimum to be (50, 50, 3)
    weights -- weights to be used by the base network, can take two values - 'None' or 'imagenet'
               'None' - random initialization of weights 
               'imagenet' - initialize imagenet weights 
    Returns:
    model -- returns the defined Keras model object
    
    
    """
    base = ResNet50(include_top=False, weights=weights, input_shape=input_shape, pooling="avg")

    # frozen layers
    for layer in base.layers[:10]:
        layer.trainable = False

    convnet_output = base.output
    
    # top layers
    convnet_output = Dense(1024)(convnet_output)
    convnet_output = BatchNormalization()(convnet_output)
    convnet_output = Activation('relu')(convnet_output)
    convnet_output = Dense(128, activation="relu")(convnet_output)
    convnet_output = Lambda(lambda x: K.l2_normalize(x, axis=1))(convnet_output)

    model = Model(inputs=base.input, outputs=convnet_output, name="base_network")

    return model


def triplet_network(base_model, input_shape=(50, 50, 3)):
    
    """
    Define triplet neural network
    
    Arguments:
    base_model -- base model to use inside the triplet network
    input_shape -- Input size of the base network, minimum to be (50, 50, 3)

    Returns:
    model -- returns the stacked triplet model object
    
    """
    # define input: query, positive, negative
    query = Input(shape=input_shape, name="query_input")
    positive = Input(shape=input_shape, name="positive_input")
    negative = Input(shape=input_shape, name="negative_input")

    # extract vector using CNN based model
    q_vec = base_model(query)
    p_vec = base_model(positive)
    n_vec = base_model(negative)

    # stack outputs
    stacks = Lambda(lambda x: K.stack(x, axis=1), name="output")([q_vec, p_vec, n_vec])

    # define the triplet model
    model = Model(inputs=[query, positive, negative], outputs=stacks, name="triplet_network")

    return model