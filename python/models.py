from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, \
    SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, UpSampling3D, Convolution3D, MaxPooling3D, \
    SpatialDropout3D, multiply, GlobalAveragePooling2D, TimeDistributed, LSTM, Layer, GlobalMaxPooling2D,\
    add, GlobalAveragePooling3D
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model, Sequential
import tensorflow as tf
from custom_layers import Mil_Attention, Last_Sigmoid
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import regularizers


def convolution_block_2d_fcn(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0, stride=1):
    x = Convolution2D(nr_of_convolutions, 1, padding='same', kernel_regularizer=l2(weight_decay),
                      strides=(stride, stride))(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if spatial_dropout:
        x = Dropout(spatial_dropout)(x)
    x = GlobalAveragePooling2D()(x)
    return x


def convolution_block_2d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0, stride=1):
    for i in range(2):
        x = Convolution2D(nr_of_convolutions, 3, padding='same', kernel_regularizer=l2(weight_decay),
                          strides=(stride, stride))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout)(x)
    return x


def convolution_block_2d_time_dist(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0, stride=1):
    for i in range(2):
        x = TimeDistributed(Convolution2D(nr_of_convolutions, 3, padding='same', kernel_regularizer=l2(weight_decay),
                                          strides=(stride, stride))(x))
        if use_bn:
            x = TimeDistributed(BatchNormalization()(x))
        x = TimeDistributed(Activation('relu')(x))
        if spatial_dropout:
            x = TimeDistributed(SpatialDropout2D(spatial_dropout)(x))
    return x


def convolution_block_3d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0, stride=1):
    for i in range(2):
        x = Convolution3D(nr_of_convolutions, 3, padding='same', kernel_regularizer=l2(weight_decay),
                          strides=(stride, stride, stride))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout3D(spatial_dropout)(x)
    return x


def encoder_block_2d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0, stride=1):
    x_before_downsampling = convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout,
                                                 weight_decay=weight_decay, stride=stride)
    x = MaxPooling2D((2, 2))(x_before_downsampling)
    return x, x_before_downsampling


def encoder_block_3d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, weight_decay=0, stride=1):
    x_before_downsampling = convolution_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout,
                                                 weight_decay=weight_decay, stride=stride)
    downsample = [2, 2, 2]
    for i in range(1, 4):
        if x.shape[i] <= 4:
            downsample[i - 1] = 1

    x = MaxPooling3D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2, weight_decay=0, stride=1):
    if dims == 2:
        return encoder_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay, stride=stride)
    elif dims == 3:
        return encoder_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay, stride=stride)
    else:
        raise ValueError


def encoder_block_time_dist(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2, weight_decay=0, stride=1):
    if dims == 2:
        return encoder_block_2d_time_dist(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay, stride=stride)
    elif dims == 3:
        return encoder_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay, stride=stride)
    else:
        raise ValueError


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2, weight_decay=0, stride=1):
    if dims == 2:
        return convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay, stride=stride)
    elif dims == 3:
        return convolution_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay, stride=stride)
    else:
        raise ValueError


def convolution_block_time_dist(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2, weight_decay=0, stride=1):
    if dims == 2:
        return convolution_block_2d_time_dist(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay, stride=stride)
    elif dims == 3:
        return convolution_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout, weight_decay=weight_decay, stride=stride)
    else:
        raise ValueError


class AttentionMIL:
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
        #while size > 4:
        for i in range(len(convolutions)):
            x, _ = encoder_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            size = int(size)
            i += 1

        # x = convolution_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout) # VGG-esque triple conv in last level

        ## define classifier model
        x = Flatten(name="flatten")(x)

        # fully-connected layers
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(
                x)
            # fc1 = BatchNormalization()(fc1)
            x = Dropout(self.dense_dropout)(x)

        alpha = Mil_Attention(L_dim=self.L_dim, output_dim=1, kernel_regularizer=l2(self.weight_decay), name='alpha',
                              use_gated=self.useGated)(x)  # L_dim=128
        x_mul = multiply([alpha, x])

        out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
        x = Model(inputs=[input_layer], outputs=[out])

        return x


class DeepMIL2D:
    def __init__(self, input_shape, nb_classes):
        #if len(input_shape) != 3:
        #    raise ValueError('Input shape must have 3 dimensions')
        #if nb_classes != 2:
        #    raise ValueError('Classes must be 2')
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

    def set_stride(self, stride):
        self.stride = stride

    def set_bn(self, use_bn):
        self.use_bn = use_bn

    def set_weight_decay(self, decay):
        self.weight_decay = decay

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
        #while size > 4:
        for i in range(len(convolutions)):
            x, _ = encoder_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            size = int(size)
            i += 1

        # x = convolution_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout) # VGG-esque triple conv in last level

        ## define classifier model
        x = Flatten(name="flatten")(x)

        # fully-connected layers
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(
                x)
            if self.use_bn:
                x = BatchNormalization()(x)
            if self.dense_dropout != None:
                x = Dropout(self.dense_dropout)(x)

        alpha = Mil_Attention(L_dim=self.L_dim, output_dim=1, kernel_regularizer=l2(self.weight_decay), name='alpha',
                              use_gated=self.useGated)(x)  # L_dim=128
        x_mul = multiply([alpha, x])

        out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
        x = Model(inputs=[input_layer], outputs=[out])

        return x


class InceptionalMIL2D:
    def __init__(self, input_shape, nb_classes):
        #if len(input_shape) != 3:
        #    raise ValueError('Input shape must have 3 dimensions')
        #if nb_classes != 2:
        #    raise ValueError('Classes must be 2')
        self.input_shape = input_shape
        self.convolutions = None
        self.use_bn = True
        self.spatial_dropout = None
        self.dense_dropout = None
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

    def set_stride(self, stride):
        self.stride = stride

    def set_bn(self, use_bn):
        self.use_bn = use_bn

    def set_weight_decay(self, decay):
        self.weight_decay = decay


    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape[1:])
        x = input_layer

        # use InceptionV3 encoder only as feature extractor -> freeze CNN weights
        #base_model = Sequential()
        #base_model.add(InceptionV3(include_top=False, weights='imagenet', input_shape=self.input_shape[1:], pooling=max))
        base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=max)
        for layer in base_model.layers:
            layer.trainable = False
        #x = base_model(x)

        x = base_model.output

        x = GlobalAveragePooling2D()(x)

        ## define classifier model
        x = Flatten(name="flatten")(x)  # (x)

        # fully-connected layers
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(
                x)
            # fc1 = BatchNormalization()(fc1)
            # x = Dropout(self.dense_dropout)(x)

        alpha = Mil_Attention(L_dim=self.L_dim, output_dim=1, kernel_regularizer=l2(self.weight_decay), name='alpha',
                              use_gated=self.useGated)(x)  # L_dim=128
        x_mul = multiply([alpha, x])

        out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
        #x = Model(inputs=[input_layer], outputs=[out])
        #x = out
        x = Model(inputs=base_model.input, outputs=out)

        return x



class VGGNet2D:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.convolutions = None
        self.use_bn = True
        self.spatial_dropout = None
        self.dense_dropout = 0.5
        self.dense_size = 64
        self.weight_decay = 0
        self.useGated = True
        self.L_dim = 32
        self.nb_dense_layers = 2
        self.final_dense = 1
        self.stride = 1

    def set_dense_size(self, size):
        self.dense_size = size

    def set_dense_dropout(self, dropout):
        self.dense_dropout = dropout

    def set_spatial_dropout(self, dropout):
        self.spatial_dropout = dropout

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions

    def set_stride(self, stride):
        self.stride = stride

    def set_bn(self, use_bn):
        self.use_bn = use_bn

    def set_weight_decay(self, decay):
        self.weight_decay = decay

    def set_final_dense(self, final_dense):
        self.final_dense = final_dense

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
        # while size > 4:
        for i in range(len(convolutions)):
            x, _ = encoder_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            size = int(size)
            i += 1

        # x = convolution_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout) # VGG-esque triple conv in last level

        ## define classifier model
        x = Flatten(name="flatten")(x)

        # fully-connected layers
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(
                x)
            # fc1 = BatchNormalization()(fc1)
            if self.use_bn:
                x = BatchNormalization()(x)
            #x = Dropout(self.dense_dropout)(x)

        x = Dense(self.nb_classes)(x)

        x = Model(inputs=input_layer, outputs=x)

        return x


class Benchline3DFCN:
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
        self.use_bn = True
        self.final_dense = 1
        self.use_output = True

    def set_dense_size(self, size):
        self.dense_size = size

    def set_dense_dropout(self, dropout):
        self.dense_dropout = dropout

    def set_spatial_dropout(self, dropout):
        self.spatial_dropout = dropout

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions

    def set_use_bn(self, use_bn):
        self.use_bn = use_bn

    def get_depth(self):
        init_size = min(self.input_shape[0], self.input_shape[1])
        size = init_size
        depth = 0
        while size > 4:
            size /= 2
            size = int(size)  # in case of odd number size before division
            depth += 1
        return depth + 1

    def set_final_dense(self, final_dense):
        self.final_dense = final_dense

    def set_use_output(self, use_output):
        self.use_output = use_output

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
        # while size > 4:
        for i in range(len(convolutions)):
            x, _ = encoder_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            size = int(size)
            i += 1

        # x = convolution_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout) # VGG-esque triple conv in last level

        '''
        ## define fully convolutional classifier layer
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Convolution2D(self.dense_size, 1, activation="relu")(x)
        #x = GlobalMaxPooling2D()(x)

        if self.use_output:
            if self.final_dense == 1:
                x = Convolution2D(self.final_dense, 1, activation="sigmoid")(x)
            else:
                x = Convolution2D(self.final_dense, 1, activation="softmax")(x)
        else:
            x = Convolution2D(self.dense_size, 1)(x)  # TODO: Note, without softmax(!)
        x = GlobalAveragePooling2D()(x)
        '''
        x = Convolution2D(self.dense_size, 1, activation="relu")(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.final_dense)(x)

        x = Model(inputs=input_layer, outputs=x)

        return x


class MLP:
    def __init__(self, input_shape, nb_classes):
        if len(input_shape) != 1:
            raise ValueError('Input shape must have 1 dimensions')
        if nb_classes != 2:
            raise ValueError('Classes must be 2')
        self.input_shape = input_shape
        self.use_bn = True
        self.dense_dropout = 0.5
        self.dense_size = 64
        self.weight_decay = 0
        self.nb_dense_layers = 2

    def set_dense_size(self, size):
        self.dense_size = size

    def set_dense_dropout(self, dropout):
        self.dense_dropout = dropout

    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        ## define classifier model

        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(x)
            # fc1 = BatchNormalization()(fc1)
            x = Dropout(self.dense_dropout)(x)

        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=input_layer, outputs=x)

        return model


class CNN3D:
    def __init__(self, input_shape, nb_classes):
        if len(input_shape) != 4:
            raise ValueError('Input shape must have 4 dimensions')
        if nb_classes != 2:
            raise ValueError('Classes must be 2')
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.convolutions = None
        self.use_bn = False
        self.spatial_dropout = None
        self.dense_dropout = 0
        self.dense_size = 64
        self.weight_decay = 0
        self.useGated = True
        self.L_dim = 32
        self.nb_dense_layers = 2
        self.stride = 1

    def set_dense_size(self, size):
        self.dense_size = size

    def set_dense_dropout(self, dropout):
        self.dense_dropout = dropout

    def set_spatial_dropout(self, dropout):
        self.spatial_dropout = dropout

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions

    def set_stride(self, stride):
        self.stride = stride

    def set_bn(self, use_bn):
        self.use_bn = use_bn

    def set_weight_decay(self, decay):
        self.weight_decay = decay

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
        # while size > 4:
        for i in range(len(convolutions)):
            x, _ = encoder_block_3d(x, convolutions[i], self.use_bn, self.spatial_dropout,
                                    weight_decay=self.weight_decay, stride=self.stride)
            size /= 2
            size = int(size)
            i += 1

        #x = convolution_block_3d(x, convolutions[i], self.use_bn, self.spatial_dropout)  # VGG-esque triple conv in last level

        # global average max pool here
        #x = GlobalAveragePooling3D()(x)

        ## define classifier model
        x = Flatten(name="flatten")(x)

        # fully-connected layers
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(
                x)
            if self.use_bn:
                x = BatchNormalization()(x)
            if self.dense_dropout != None:
                x = Dropout(self.dense_dropout)(x)

        x = Dense(self.nb_classes, activation="softmax")(x)
        x = Model(inputs=input_layer, outputs=x)

        return x


'''
# Say you have a backbone model works for an input of (16,112,112,3) and you want to use it under Multiple instance
# learning. You just need to wrap it with TimeDistributed and use some kind of 'selector' to pick the salient result.
# Here's an example when there're 32 instances in each bag.

input = Input(shape=(32,16,112,112,3))
x = BatchNormalization()(input)
x = TimeDistributed(backbone)(x)
x = GlobalMaxPool1D()(x)
model = Model(input,x)
'''

class DeepMIL2D_hybrid:
    def __init__(self, input_shape, nb_classes):  # input_shape = (bag_size, ) + data_size + (nb_channels,) # data size could be (16, 112, 112) for 3D slabs
        if len(input_shape) != 4:
            raise ValueError('Input shape must have 4 dimensions')
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
        x = BatchNormalization()(x)  # <- batch norm over the data first?
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
        #while size > 4:
        for i in range(len(convolutions)):
            x, _ = encoder_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            size = int(size)
            i += 1

        # x = convolution_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout) # VGG-esque triple conv in last level

        ## define classifier model
        x = Flatten(name="flatten")(x)

        # fully-connected layers
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(x)
            # fc1 = BatchNormalization()(fc1)
            x = Dropout(self.dense_dropout)(x)

        y = BatchNormalization()(input_layer)
        x = TimeDistributed(x)(y)
        x = GlobalMaxPool1D()(x)
        model = Model(input_layer, x)

        return model


class CNNLSTM2D:
    def __init__(self, input_shape, nb_classes):  # input_shape = (bag_size, ) + data_size + (nb_channels,) # data size could be (16, 112, 112) for 3D slabs
        if len(input_shape) != 4:
            raise ValueError('Input shape must have 4 dimensions')
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
        #while size > 4:
        for i in range(len(convolutions)):
            x, _ = encoder_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            size = int(size)
            i += 1

        # x = convolution_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout) # VGG-esque triple conv in last level

        ## define classifier model
        x = Flatten(name="flatten")(x)

        # fully-connected layers
        for i, d in enumerate(range(self.nb_dense_layers)):
            x = Dense(self.dense_size, activation='relu', kernel_regularizer=l2(self.weight_decay), name="fc" + str(i))(x)
            # fc1 = BatchNormalization()(fc1)
            x = Dropout(self.dense_dropout)(x)

        model = Sequential()
        model.add(TimeDistributed(x))
        model.add(LSTM())
        model.add(Dense(self.dense_size))

        return model





# https://github.com/dancsalo/TensorFlow-MIL/blob/master/deep_mil.py
class NoisyAnd(Layer):
    """Custom NoisyAND layer from the Deep MIL paper"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = 10  # fixed, controls the slope of the activation
        #print(input_shape)
        #print(input_shape[3])
        self.b = self.add_weight(name='b', shape=(1, self.output_dim), initializer='uniform', trainable=True)  # input_shape[3]), initializer='uniform', trainable=True)
        super(NoisyAnd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        res = (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (
                tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


# https://github.com/dancsalo/TensorFlow-MIL/blob/master/deep_mil.py
def DeepFCNMIL(input_shape, nb_classes):
    """Define Deep FCN for MIL, layer-by-layer from original paper"""
    model = Sequential()
    print(input_shape)
    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(Convolution2D(1000, (3, 3), activation='relu'))
    model.add(Convolution2D(nb_classes, (1, 1), activation='relu'))
    model.add(NoisyAnd(nb_classes))
    model.add(Dense(nb_classes, activation="sigmoid"))  # activation='softmax'))
    return model


def model2D():  # TODO: Make this a class...
    data_input = Input(shape=input_shape[1:] + (1,), dtype='float32', name='input')
    conv1 = Convolution2D(16, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(data_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(16, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Convolution2D(32, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(32, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Convolution2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = MaxPooling2D((2, 2))(conv4)

    conv5 = Convolution2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv5)
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

def residual_block(prev_layer, repetitions, num_filters):
    block = prev_layer
    for i in range(repetitions):
        x = Convolution2D(
                filters=num_filters,
                kernel_size=3,
                #kernel_regularizer=regularizers.l2(1e-2),
                #bias_regularizer=regularizers.l2(1e-2),
                strides=1,
                padding='same'
        )(block)
        #x = Activation('relu')(x)
        y = Activation('relu')(x)
        #y = BatchNormalization()(x)
        #y = BatchNormalization()(y)  # TODO: INTRODUCED BATCHNORM HERE -> resulted in high conf on class 1 only (float16 issue with batchnorm?)
        x = Convolution2D(
                filters=num_filters,
                kernel_size=3,
                #kernel_regularizer=regularizers.l2(1e-2),  # TODO: Tried to remove L2-regularization
                #bias_regularizer=regularizers.l2(1e-2),  # TODO: Tried to remove L2-regularization
                strides=1,
                padding='same'
        )(x)
        x = add([x, y])
        x = Activation('relu')(x)
        #x = BatchNormalization()(x)
        #x = BatchNormalization()(x)  # TODO: INTRODUCED BATCHNORM HERE
        block = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    return block


def resnet(input_shape, num_classes, ds=2):  # num_classes, num_channels=1, ds=2):

    inputs = Input(shape=input_shape)  # (None, None, num_channels))

    x = Convolution2D(
        filters=64//ds,
        kernel_size=7,
        #kernel_regularizer=regularizers.l2(1e-2),  # TODO: Tried to remove L2-regularization
        #bias_regularizer=regularizers.l2(1e-2),  # TODO: Tried to remove L2-regularization
        strides=2,
        padding='same'
    )(inputs)
    block_0 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    block_1 = residual_block(prev_layer=block_0, repetitions=3, num_filters=64//ds)
    block_2 = residual_block(prev_layer=block_1, repetitions=4, num_filters=128//ds)
    block_3 = residual_block(prev_layer=block_2, repetitions=6, num_filters=256//ds)
    block_4 = residual_block(prev_layer=block_3, repetitions=3, num_filters=512//ds)

    x = GlobalAveragePooling2D()(block_4)
    outputs = Dense(num_classes)(x)  # , activation="softmax")(x)  # TODO: Forgot sigmoid activation here in implementation??? No. softmax is applied in step_bag_gradient in train
    #outputs = Dense(num_classes, dtype='float32')(x)  # TODO: Perhaps need to cast to float32 in last layer before softmax?

    model = Model(inputs=inputs, outputs=outputs)

    return model


