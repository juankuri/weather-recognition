import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.applications import EfficientNetB0

# Set random seed for TensorFlow
tf.random.set_seed(11)
path = './dataset/'

image_shape = (224,224,3)
batch_size = 64
epochs = 100
no_classes = 11

train_path = os.path.join(path, 'train')
test_path = os.path.join(path, 'test')

train_image_gen = ImageDataGenerator(rescale=1/255,            # rescale the image by normalzing it
                                    validation_split=0.2)      # split the data

test_image_gen = ImageDataGenerator(rescale=1/255)             # rescale the image by normalzing it
          
# train_image_gen = ImageDataGenerator(rescale=1/255,validation_split=0.2,   rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.1,horizontal_flip=True)
# test_image_gen = ImageDataGenerator(    rescale=1/255)

# Defining training dataset
train_image_ds = train_image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               subset = 'training')

# Defining validation dataset
val_image_ds = train_image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               subset = 'validation')


# Load pre-trained EfficientNet-B5 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(11, activation='softmax')(x) 
EffB0_model = Model(inputs=base_model.input, outputs=predictions)

early_stop = EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights=True )

# Compile the model with SGD optimizer
sgd = SGD(0.01, momentum=0.9, nesterov=True)
EffB0_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

EffB0_model_history = EffB0_model.fit(train_image_ds,
                    epochs=epochs,               
                    validation_data=val_image_ds,
                    callbacks=[early_stop])

EffB0_model.save("weather_model/1")
