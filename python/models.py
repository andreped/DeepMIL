from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, UpSampling3D, Convolution3D, MaxPooling3D, SpatialDropout3D, multiply
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model
import tensorflow as tf
from custom_layers import Mil_Attention, Last_Sigmoid


def convolution_block_2d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0):
    for i in range(2):
        x = Convolution2D(nr_of_convolutions, 3, padding='same', kernel_regularizer=l2(weight_decay))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout)(x)
    return x

def encoder_block_2d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0):
    x_before_downsampling = convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay)
    x = MaxPooling2D((2, 2))(x_before_downsampling)
    return x, x_before_downsampling


def convolution_block_3d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0):
    for i in range(2):
        x = Convolution3D(nr_of_convolutions, 3, padding='same', weight_decay=weight_decay)(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout3D(spatial_dropout)(x)

    return x

def encoder_block_3d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0):

    x_before_downsampling = convolution_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay)
    downsample = [2, 2, 2]
    for i in range(1, 4):
        if x.shape[i] <= 4:
            downsample[i-1] = 1

    x = MaxPooling3D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2, weight_decay=0):
    if dims == 2:
        return encoder_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay)
    elif dims == 3:
        return encoder_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay)
    else:
        raise ValueError


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2, weight_decay=0):
    if dims == 2:
        return convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay)
    elif dims == 3:
        return convolution_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay)
    else:
        raise ValueError


class DeepMIL2D():
    def __init__(self, input_shape, nb_classes):
        if len(input_shape) != 3:
            raise ValueError('Input shape must have 3 dimensions')
        if nb_classes != 2:
            raise ValueError('Classes must be 2')
        self.input_shape = input_shape
        self.convolutions = None
        self.use_bn = True
        self.spatial_dropout = None
        self.dense_dropout = 0.5
        self.dense_size = 64
        self.weight_decay = 0
        self.useGated = True
        self.L_dim = 32
        self.nb_dense_layers = 2

    def set_dense_size(self, size):
        self.dense_size = size

    def set_dense_dropout(self, dropout):
        self.dense_dropout = dropout

    def set_spatial_dropout(self, dropout):
        self.spatial_dropout = dropout

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions

    def get_depth(self):
        init_size = min(self.input_shape[0], self.input_shape[1])
        size = init_size
        depth = 0
        while size > 4:
            size /= 2
            size = int(size)  # in case of odd number size before division
            depth += 1
        return depth + 1

    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        init_size = min(self.input_shape[:-1])
        size = init_size

        convolutions = self.convolutions
        if convolutions is None:  # if not defined, define simple encoder
            # Create convolutions
            convolutions = []
            nr_of_convolutions = 8
            for i in range(self.get_depth()):
                convolutions.append(nr_of_convolutions)
                nr_of_convolutions *= 2

        ## make encoder
        i = 0
        while size > 4:
            x, _ = encoder_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            size = int(size)
            i += 1

        #x = convolution_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout) # VGG-esque triple conv in last level

        ## define classifier model
        x = Flatten(name="flatten")(x)

        # fully-connected layers
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(x)
            # fc1 = BatchNormalization()(fc1)
            x = Dropout(self.dense_dropout)(x)

        alpha = Mil_Attention(L_dim=self.L_dim, output_dim=1, kernel_regularizer=l2(self.weight_decay), name='alpha',
                              use_gated=self.useGated)(x)  # L_dim=128
        x_mul = multiply([alpha, x])

        out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
        x = Model(inputs=[input_layer], outputs=[out])

        return x



def model2D():  # TODO: Make this a class...
    data_input = Input(shape=input_shape[1:] + (1,), dtype='float32', name='input')
    conv1 = Conv2D(16, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(data_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = MaxPooling2D((2, 2))(conv5)

    x = Flatten()(conv5)

    # fully-connected layers
    fc1 = Dense(64, activation='relu', kernel_regularizer=l2(weight_decay), name='fc1')(x)
    # fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(64, activation='relu', kernel_regularizer=l2(weight_decay), name="fc2")(fc1)
    # fc2 = BatchNormalization()(fc2)
    fc2 = Dropout(0.5)(fc2)

    alpha = Mil_Attention(L_dim=32, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha',
                          use_gated=useGated)(fc2)  # L_dim=128
    x_mul = multiply([alpha, fc2])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
    model = Model(inputs=[data_input], outputs=[out])

    return model
