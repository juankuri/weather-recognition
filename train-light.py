import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.applications import MobileNetV2
from keras import mixed_precision       # <<< SOLO ESTA IMPORTACIÓN

# ===========================
# MIXED PRECISION (nuevo API)
# ===========================
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)
print("Mixed precision:", policy)


# ===========================
# CONFIGURACIÓN BASE
# ===========================
tf.random.set_seed(11)

path = './dataset/'

image_shape = (160,160,3)
batch_size = 64
epochs = 20
no_classes = 11

train_path = os.path.join(path, 'train')
test_path = os.path.join(path, 'test')

# ===========================
# DATA GENERATORS
# ===========================
train_image_gen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

test_image_gen = ImageDataGenerator(rescale=1/255)

train_image_ds = train_image_gen.flow_from_directory(
    train_path,
    target_size=image_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_image_ds = train_image_gen.flow_from_directory(
    train_path,
    target_size=image_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Convertir a tf.data
train_ds = tf.data.Dataset.from_generator(
    lambda: train_image_ds,
    output_signature=(
        tf.TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 11), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: val_image_ds,
    output_signature=(
        tf.TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 11), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)


# ===========================
# MODELO
# ===========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=image_shape
)

# Freeze inicial
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)

# IMPORTANTE: salida en float32 aunque mixed precision esté activo
predictions = Dense(no_classes, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

sgd = SGD(0.01, momentum=0.9, nesterov=True)

model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===========================
# TRAIN 1
# ===========================
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[early_stop]
)

# ===========================
# FINETUNE
# ===========================
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=SGD(0.001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)
