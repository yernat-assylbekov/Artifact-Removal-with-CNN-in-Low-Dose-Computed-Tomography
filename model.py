import tensorflow as tf

from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU, Conv2DTranspose, Input, concatenate
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.math import reduce_mean, square, squared_difference, log, negative

def SNR(Y_true, Y_pred):
    signal = reduce_mean(square(Y_true))
    noise = reduce_mean(squared_difference(Y_pred, Y_true))

    return 10 * log(signal / noise) / log(10.0)

def neg_SNR(Y_true, Y_pred):
    snr = SNR(Y_true, Y_pred)

    return negative(snr)

def Modified_U_Net(input_size, input_channels, filters, learning_rate, scale):

    # set up input
    X = Input(shape=(input_size, input_size, input_channels))
    Y = X

    # convolution pre-processing
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    # convolutions 1
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y_skip_1 = BatchNormalization()(Y)

    # pooling 1
    Y = MaxPool2D(pool_size=2, strides=2, padding='valid')(Y_skip_1)

    # convolutions 2
    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y_skip_2 = BatchNormalization()(Y)

    # pooling 2
    Y = MaxPool2D(pool_size=2, strides=2, padding='valid')(Y_skip_2)

    # convolutions 3
    Y = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y_skip_3 = BatchNormalization()(Y)

    # pooling 3
    Y = MaxPool2D(pool_size=2, strides=2, padding='valid')(Y_skip_3)

    # convolutions 4
    Y = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y_skip_4 = BatchNormalization()(Y)

    # pooling 4
    Y = MaxPool2D(pool_size=2, strides=2, padding='valid')(Y_skip_4)

    # convolutions 5
    Y = Conv2D(filters=16*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=16*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    # upconvolution and concatenation 1
    Y = Conv2DTranspose(filters=8*filters, kernel_size=2, strides=2, padding='valid', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = concatenate([Y_skip_4, Y])

    # convolutions 6
    Y = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    # upconvolution and concatenation 2
    Y = Conv2DTranspose(filters=4*filters, kernel_size=2, strides=2, padding='valid', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = concatenate([Y_skip_3, Y])

    # convolutions 7
    Y = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    # upconvolution and concatenation 3
    Y = Conv2DTranspose(filters=2*filters, kernel_size=2, strides=2, padding='valid', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = concatenate([Y_skip_2, Y])

    # convolutions 8
    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    # upconvolution and concatenation 4
    Y = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='valid', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = concatenate([Y_skip_1, Y])

    # convolutions 9
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    # convolution post-processing
    Y = Conv2D(filters=input_channels, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)
    Y = BatchNormalization()(Y)

    # final concatenation and convolution
    Y = concatenate([X, Y])
    Y = Conv2D(filters=input_channels, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(l=scale), bias_regularizer=l2(l=scale))(Y)

    # set up model and compile
    model = Model(inputs=X, outputs=Y)
    model.compile(optimizer=Adam(lr=learning_rate), loss=neg_SNR, metrics=[SNR])

    # return model
    return model
