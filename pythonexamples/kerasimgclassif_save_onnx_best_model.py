#https://keras.io/examples/cifar10_cnn/
# https://www.geeksforgeeks.org/python-image-classification-using-keras/
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx
import onnxruntime
import tensorflow.keras.models as keras_models

import onnx
import keras2onnx

best_model = keras_models.load_model('.mdl_wts.hdf5')
onnx_model = keras2onnx.convert_keras(best_model, best_model.name)
#save model
temp_model_file = 'modelkerasimgBEST.onnx'
onnx.save_model(onnx_model, temp_model_file)


temp_model_file = '../nodejsimgclassification/public/modelkerasimgBEST.onnx'
onnx.save_model(onnx_model, temp_model_file)