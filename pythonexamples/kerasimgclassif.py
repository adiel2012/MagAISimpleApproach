#https://keras.io/examples/cifar10_cnn/
# https://www.geeksforgeeks.org/python-image-classification-using-keras/

from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K   
from keras.callbacks.callbacks import EarlyStopping  
from keras.callbacks import ModelCheckpoint
from keras.callbacks.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.engine import InputLayer
import keras2onnx
import onnxruntime
import tensorflow.keras.models as keras_models
import onnx
from keras.engine.training import Model
from keras.engine.input_layer import Input

  
img_width, img_height = 224, 224
  
train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples = 400 
nb_validation_samples = 100
epochs = 25
batch_size = 16
  
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
  
model = Sequential() 
model.add(InputLayer(input_shape=(img_width, img_height, 3) , name="input_1"))
model.add(Conv2D(32, (3, 3), padding='same', input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
  
# initiate RMSprop optimizer
opt = RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
  
train_datagen = ImageDataGenerator( 
                rescale = 1. / 255, 
                 shear_range = 0.2, 
                  zoom_range = 0.2, 
            horizontal_flip = True) 
  
test_datagen = ImageDataGenerator(rescale = 1. / 255) 
  
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='categorical') 
  
validation_generator = test_datagen.flow_from_directory( 
                                    validation_data_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='categorical') 

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
  
model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = nb_validation_samples // batch_size
    ,callbacks=[earlyStopping, mcp_save]) 

model._layers.pop(0)
newInput = Input(batch_shape=(1,224,224,3))   
newOutputs = model(newInput)
model = Model(newInput, newOutputs)
model.save_weights('model_saved.h5') 
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = 'modelkerasimg.onnx'
onnx.save_model(onnx_model, temp_model_file)

model = mcp_save.model

model._layers.pop(0)
newInput = Input(batch_shape=(1,224,224,3), name="input_1")   
newOutputs = model(newInput)
model = Model(newInput, newOutputs)
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = 'modelkerasimgBEST.onnx'
onnx.save_model(onnx_model, temp_model_file)
temp_model_file = '../nodejsimgclassification/public/modelkerasimgBEST.onnx'
onnx.save_model(onnx_model, temp_model_file)