import keras
from keras.models import Model
from keras.layers import *

def custom_linear_2(x):
    return keras.activations.linear(x, )
def custom_relu_alpha_0__max_value_None11(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None10(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_linear_(x):
    return keras.activations.linear(x, )
def custom_relu_alpha_0__max_value_None9(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None8(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None7(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None6(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None5(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None4(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None3(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None2(x):
    return keras.activations.relu(x, alpha=0, max_value=None)
def custom_relu_alpha_0__max_value_None(x):
    return keras.activations.relu(x, alpha=0, max_value=None)

input2 = Input(shape=(,,3), batch_shape=None, dtype=None, sparse=False, tensor=None)
input3 = Input(shape=(,,3), batch_shape=None, dtype=None, sparse=False, tensor=None)
concatenate = Concatenate(axis=3)(inputs=[input3, input2])
conv2d = Conv2D(filters=16, kernel_size=7, strides=1, padding="valid", data_format=None, activation=custom_relu_alpha_0__max_value_None, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=concatenate)
conv2d2 = Conv2D(filters=32, kernel_size=5, strides=2, padding="valid", data_format=None, activation=custom_relu_alpha_0__max_value_None2, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d)
conv2d3 = Conv2D(filters=64, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_relu_alpha_0__max_value_None3, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d2)
conv2d4 = Conv2D(filters=128, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_relu_alpha_0__max_value_None4, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d3)
conv2d5 = Conv2D(filters=256, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_relu_alpha_0__max_value_None5, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d4)
conv2d6 = Conv2D(filters=256, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_relu_alpha_0__max_value_None6, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d5)
conv2d7 = Conv2D(filters=512, kernel_size=3, strides=2, padding="valid", data_format=None, activation=custom_relu_alpha_0__max_value_None7, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=conv2d6)
flatten = Flatten()(inputs=conv2d7)
dense = Dense(units=512, activation=custom_relu_alpha_0__max_value_None8, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=flatten)
dense2 = Dense(units=512, activation=custom_relu_alpha_0__max_value_None9, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=dense)
dense3 = Dense(units=3, activation=custom_linear_, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=dense2)
dense4 = Dense(units=512, activation=custom_relu_alpha_0__max_value_None10, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=flatten)
dense5 = Dense(units=512, activation=custom_relu_alpha_0__max_value_None11, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=dense4)
dense6 = Dense(units=3, activation=custom_linear_2, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())(inputs=dense5)

custom_objects = {}
custom_objects['custom_relu_alpha_0__max_value_None'] = custom_relu_alpha_0__max_value_None
custom_objects['custom_relu_alpha_0__max_value_None2'] = custom_relu_alpha_0__max_value_None2
custom_objects['custom_relu_alpha_0__max_value_None3'] = custom_relu_alpha_0__max_value_None3
custom_objects['custom_relu_alpha_0__max_value_None4'] = custom_relu_alpha_0__max_value_None4
custom_objects['custom_relu_alpha_0__max_value_None5'] = custom_relu_alpha_0__max_value_None5
custom_objects['custom_relu_alpha_0__max_value_None6'] = custom_relu_alpha_0__max_value_None6
custom_objects['custom_relu_alpha_0__max_value_None7'] = custom_relu_alpha_0__max_value_None7
custom_objects['custom_relu_alpha_0__max_value_None8'] = custom_relu_alpha_0__max_value_None8
custom_objects['custom_relu_alpha_0__max_value_None9'] = custom_relu_alpha_0__max_value_None9
custom_objects['custom_linear_'] = custom_linear_
custom_objects['custom_relu_alpha_0__max_value_None10'] = custom_relu_alpha_0__max_value_None10
custom_objects['custom_relu_alpha_0__max_value_None11'] = custom_relu_alpha_0__max_value_None11
custom_objects['custom_linear_2'] = custom_linear_2


model = Model(inputs=[input3,input2], outputs=[dense6,dense3])
result = model
model.custom_objects = custom_objects