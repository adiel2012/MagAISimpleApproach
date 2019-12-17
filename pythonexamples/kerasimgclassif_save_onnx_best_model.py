#https://keras.io/examples/cifar10_cnn/
# https://www.geeksforgeeks.org/python-image-classification-using-keras/
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx
import onnxruntime
import tensorflow.keras.models as keras_models

import onnx
import keras2onnx
from keras.engine import InputLayer

from keras.engine.training import Model
from keras.engine.input_layer import Input

model = keras_models.load_model('.mdl_wts.hdf5')


model.layers[0] = InputLayer(input_shape=(1,224,224,3), name="input_1")

#model._layers.pop(0)
#newInput = Input(shape=(None,224,224,3))    # let us say this new InputLayer
#newOutputs = model(newInput)
#model = Model(newInput, newOutputs)

onnx_model = keras2onnx.convert_keras(model, model.name)
#save model
temp_model_file = 'modelkerasimgBEST.onnx'
onnx.save_model(onnx_model, temp_model_file)


temp_model_file = '../nodejsimgclassification/public/modelkerasimgBEST.onnx'
onnx.save_model(onnx_model, temp_model_file)