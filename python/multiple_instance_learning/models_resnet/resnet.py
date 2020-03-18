from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    add,
    Dropout,
    Activation,
    Dense,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def residual_block(prev_layer, repetitions, num_filters):
    block = prev_layer
    for i in range(repetitions):
        x = Conv2D(
                filters=num_filters,
                kernel_size=3, 
                kernel_regularizer=regularizers.l2(1e-2), 
                bias_regularizer=regularizers.l2(1e-2), 
                strides=1, 
                padding='same'
        )(block)
        #x = Activation('relu')(x)
        y = Activation('relu')(x)
        #y = BatchNormalization()(x)
        #y = BatchNormalization()(y)  # TODO: INTRODUCED BATCHNORM HERE -> resulted in high conf on class 1 only (float16 issue with batchnorm?)
        x = Conv2D(
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

    x = Conv2D(
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

    model = Model(inputs=inputs, outputs=outputs)

    return model
