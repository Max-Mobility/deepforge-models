import keras
from keras.models import Model
from keras.layers import *

def custom_sigmoid_(x):
    return keras.activations.sigmoid(x, )
def custom_elu_alpha_114(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_113(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_112(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_111(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_110(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_19(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_18(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_17(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_16(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_15(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_14(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_13(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_12(x):
    return keras.activations.elu(x, alpha=1)
def custom_elu_alpha_1(x):
    return keras.activations.elu(x, alpha=1)

input2 = Input(shape=(,,3), batch_shape=None, dtype=None, sparse=False, tensor=None)
conv2d = Conv2D(filters=32, kernel_size=7, strides=1, padding="valid", data_format=None, activation=custom_elu_alpha_1, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=input2)
conv2d2 = Conv2D(filters=32, kernel_size=7, strides=2, padding="valid", data_format=None, activation=custom_elu_alpha_12, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d)
conv2d3 = Conv2D(filters=64, kernel_size=5, strides=1, padding="valid", data_format=None, activation=custom_elu_alpha_13, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d2)
conv2d4 = Conv2D(filters=64, kernel_size=5, strides=2, padding="valid", data_format=None, activation=custom_elu_alpha_14, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d3)
conv2d5 = Conv2D(filters=128, kernel_size=3, strides=1, padding="valid", data_format=None, activation=custom_elu_alpha_15, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d4)
conv2d6 = Conv2D(filters=128, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_elu_alpha_16, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d5)
conv2d7 = Conv2D(filters=256, kernel_size=3, strides=1, padding="valid", data_format=None, activation=custom_elu_alpha_17, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d6)
conv2d8 = Conv2D(filters=256, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_elu_alpha_18, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d7)
conv2d9 = Conv2D(filters=512, kernel_size=3, strides=1, padding="valid", data_format=None, activation=custom_elu_alpha_19, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d8)
conv2d10 = Conv2D(filters=512, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_elu_alpha_110, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d9)
conv2d11 = Conv2D(filters=512, kernel_size=3, strides=1, padding="valid", data_format=None, activation=custom_elu_alpha_111, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d10)
conv2d12 = Conv2D(filters=512, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_elu_alpha_112, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d11)
conv2d13 = Conv2D(filters=512, kernel_size=3, strides=1, padding="valid", data_format=None, activation=custom_elu_alpha_113, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d12)
conv2d14 = Conv2D(filters=512, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_elu_alpha_114, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d13)
conv2dtranspose = Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding="same", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d14)
concatenate = Concatenate(axis=3)(inputs=[conv2dtranspose, conv2d12])
conv2dtranspose2 = Conv2DTranspose(filters=512, kernel_size=3, strides=1, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=concatenate)
conv2dtranspose3 = Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2dtranspose2)
concatenate2 = Concatenate(axis=3)(inputs=[conv2dtranspose3, conv2d10])
conv2dtranspose4 = Conv2DTranspose(filters=512, kernel_size=3, strides=1, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=concatenate2)
conv2dtranspose5 = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2dtranspose4)
concatenate3 = Concatenate(axis=3)(inputs=[conv2dtranspose5, conv2d8])
conv2dtranspose6 = Conv2DTranspose(filters=256, kernel_size=3, strides=1, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=concatenate3)
conv2dtranspose7 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2dtranspose6)
concatenate4 = Concatenate(axis=3)(inputs=[conv2dtranspose7, conv2d6])
conv2dtranspose8 = Conv2DTranspose(filters=128, kernel_size=3, strides=1, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=concatenate4)
conv2dtranspose9 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2dtranspose8)
concatenate5 = Concatenate(axis=3)(inputs=[conv2dtranspose9, conv2d4])
conv2dtranspose10 = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=concatenate5)
conv2dtranspose11 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2dtranspose10)
concatenate6 = Concatenate(axis=3)(inputs=[conv2dtranspose11, conv2d2])
conv2dtranspose12 = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=concatenate6)
conv2dtranspose13 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2dtranspose12)
conv2dtranspose14 = Conv2DTranspose(filters=16, kernel_size=3, strides=1, padding="valid", data_format=None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2dtranspose13)
conv2d15 = Conv2D(filters=2, kernel_size=3, strides=1, padding="valid", data_format=None, activation=custom_sigmoid_, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2dtranspose14)

custom_objects = {}
custom_objects['custom_elu_alpha_1'] = custom_elu_alpha_1
custom_objects['custom_elu_alpha_12'] = custom_elu_alpha_12
custom_objects['custom_elu_alpha_13'] = custom_elu_alpha_13
custom_objects['custom_elu_alpha_14'] = custom_elu_alpha_14
custom_objects['custom_elu_alpha_15'] = custom_elu_alpha_15
custom_objects['custom_elu_alpha_16'] = custom_elu_alpha_16
custom_objects['custom_elu_alpha_17'] = custom_elu_alpha_17
custom_objects['custom_elu_alpha_18'] = custom_elu_alpha_18
custom_objects['custom_elu_alpha_19'] = custom_elu_alpha_19
custom_objects['custom_elu_alpha_110'] = custom_elu_alpha_110
custom_objects['custom_elu_alpha_111'] = custom_elu_alpha_111
custom_objects['custom_elu_alpha_112'] = custom_elu_alpha_112
custom_objects['custom_elu_alpha_113'] = custom_elu_alpha_113
custom_objects['custom_elu_alpha_114'] = custom_elu_alpha_114
custom_objects['custom_sigmoid_'] = custom_sigmoid_


model = Model(inputs=[input2], outputs=[conv2d15])
result = model
model.custom_objects = custom_objects