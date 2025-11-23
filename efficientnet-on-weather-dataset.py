#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# - Weather recognition is an important task in various fields such as agriculture, transportation, and disaster control. 
# - The ability to accurately classify weather conditions from images can help in making  decisions and mitigating risks associated with weather related events. 
# - This problem statement aims to develop a multi-class classification model capable of accurately identifying different weather conditions including dew, fog smog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, and snow from images.

# # Data Definition
# 
# The dataset consists of 11 folders, each representing a different weather condition. The folders and their respective file counts are as follows:
# 
# 1.  `dew - 698 files`
# 2.  `fogsmog - 851 files`
# 3.  `frost - 475 files`
# 4.  `glaze - 639 files`
# 5.  `hail - 591 files`
# 6.  `lightning - 377 files`
# 7.  `rain - 526 files`
# 8.  `rainbow - 232 files`
# 9.  `rime - 1160 files`
# 10. `sandstorm - 692 files`
# 11. `snow - 621 files`
# 
# 
# Each file within the folders contains an image corresponding to the respective weather condition. These images will serve as the input data for training and evaluating the multi-class classification model.

# # Import Libraries

# In[ ]:


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dropout, BatchNormalization 
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten,GlobalAveragePooling2D, Reshape, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.utils import array_to_img 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# In[80]:


from keras.optimizers import Adam ,Adamax , SGD
from keras.applications import EfficientNetB0, EfficientNetB4

# In[81]:


# Set random seed for TensorFlow
tf.random.set_seed(42)

# # Data Preview 
# Let's take a look at the images in the dataset and check thier
# * Size
# * Number of channels

# In[4]:


path = '/kaggle/input/weather-dataset/dataset'

# In[5]:


# We have 11 different weather conditions
os.listdir(path)

# In[6]:


fig = plt.figure(figsize=(6,6)) 
rows = 3
columns = 4


for i, image_filename in enumerate(os.listdir(path)):

    first_img_path = os.listdir(os.path.join(path, image_filename))[0]
    first_image = imread(os.path.join(path, image_filename, first_img_path))
    fig.add_subplot(rows, columns, i+1) 
    plt.axis('off') 
    plt.imshow(first_image)
    plt.title('{}'.format(image_filename, fontsize=10))

plt.tight_layout()
plt.show()

# **Let's find out the average dimensions of these images.**

# In[7]:


dim1 = []    # width
dim2 = []    # height
colors = []  # color channel

for i, image_filename in enumerate(os.listdir(path)):
    
    subfolders_path = os.path.join(path, image_filename)
    print(subfolders_path)
    
    for image_path in os.listdir(subfolders_path):
        
        image = imread(os.path.join(subfolders_path,image_path))

        # handling gray scale images if any
        if len(image.shape) < 3:
            image = image.reshape(image.shape+(1,))
            
        d1,d2,color = image.shape
        dim1.append(d1)
        dim2.append(d2)
        colors.append(color)

# In[8]:


# There is a wide variety of image dimensions in the dataset
sns.scatterplot(x=dim1, y=dim2, alpha = 0.7)
plt.xlabel('Image Widths')
plt.ylabel('Image Heights')
plt.title('Widths vs Heights');

# 
# - The image widths reach up to 3000.
# - The image heights reach up to 5000.

# In[9]:


# mean of images width
np.mean(dim1)

# In[10]:


# mean of images height
np.mean(dim2)

# In[11]:


# number of color channels found
np.unique(colors)

# The number of color channels found in the images are 1, 3 and 4 :
# 
# - "1" : Indicates grayscale images, which have a single color channel.
# - "3" : Represents RGB (Red, Green, Blue) images, where each pixel is represented by three color channels.
# - "4" : Indicates RGBA (Red, Green, Blue, Alpha) images, which include an additional alpha channel for transparency or opacity information.
# 

#  - **It's beneficial for the input images to have the same size when training a CNN. Therefore we could specify the input_shape to be (373,520,3).**

# In[12]:


# mean of all images dimensions
input_shape =  (373,520,3)

# In[13]:


all_data_sum=0
images_count_dict = {}
for image_filename in os.listdir(path):

    folder_path = os.path.join(path, image_filename)
    images_count = len(os.listdir(folder_path))
    images_count_dict[image_filename] = images_count
    all_data_sum+= images_count
    print('{} folder has {} images'.format(image_filename ,  images_count))

print("\nTotal Number of Images: {} image".format(all_data_sum))

# In[14]:


data = list(images_count_dict.items())

# Create a bar plot using Seaborn
plt.figure(figsize=(8, 5))
sns.barplot(y=[x[0] for x in data], x=[x[1] for x in data], palette='crest',orient='horizontal')
plt.ylabel('Class Labels')
plt.xlabel('Number of Images')
plt.title('Distribution of Class Labels');

#  ## Problem
# **Since we have an imbalanced dataset and relatively small dataset, we need to consider taking some steps, such as:**
# - Collecting more data
# - Data Augmentation
# - Choosing appropriate evaluation metrics that are robust to class imbalances. 

# ## Solution
# - **Data collection**: 
#      - We have collected more data from another datasets, such as, [link]https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset/data.
# 
# - **Data Preprocessing**:
#    - We've utilized data augmentation techniques to increase the number of data and balance it. By applying various transformations to the existing data, such as rotation, shifting, flipping, and zooming, data augmentation helped create additional training examples.
#    
# ### ImageDataGenerator Used Parameters:
# 
# - `rotation_range`: Random rotation applied to the images within 20 degrees.
# - `width_shift_range`: Random horizontal shift applied to the images as a fraction of total width (10%).
# - `height_shift_range`: Random vertical shift applied to the images as a fraction of total height (10%).
# - `brightness_range`: Range for randomly adjusting brightness of the images from the range [0.5-1.5].
# - `shear_range`: Random shear intensity applied to the images (15%).
# - `zoom_range`: Range for randomly zooming into the images (20%).
# - `horizontal_flip`: Randomly flips images horizontally.
# - `fill_mode`: Strategy for filling in newly created pixels.
# - `rescale`: Rescaling factor for pixel values normalization.
# 
# These parameters are applied to augment the training data and enhance the robustness and generalization of the model.
# 

# In[15]:


# paths to new dataset
train_path = '/kaggle/input/new-weather-ds/new dataset/train'
test_path = '/kaggle/input/new-weather-ds/new dataset/test'

# In[16]:


# We have 11 different weather conditions
os.listdir(train_path)

# In[17]:


os.listdir(test_path)

# ## Augmented data preview

# In[18]:


fig = plt.figure(figsize=(6,6)) 
rows = 3
columns = 4


for i, image_filename in enumerate(os.listdir(train_path)):

    first_img_path = os.listdir(os.path.join(train_path, image_filename))[0]
    first_image = imread(os.path.join(train_path, image_filename, first_img_path))
    fig.add_subplot(rows, columns, i+1) 
    plt.axis('off') 
    plt.imshow(first_image)
    plt.title('{}'.format(image_filename, fontsize=10))

plt.tight_layout()
plt.show()

# In[19]:


all_data_sum=0
images_count_dict = {}
for image_filename in os.listdir(train_path):

    folder_path = os.path.join(train_path, image_filename)
    images_count = len(os.listdir(folder_path))
    images_count_dict[image_filename] = images_count
    all_data_sum+= images_count
    print('{} folder has {} images'.format(image_filename ,  images_count))

print("\nTotal Number of Images: {} image".format(all_data_sum))


# In[20]:


data = list(images_count_dict.items())

# Create a bar plot using Seaborn
plt.figure(figsize=(8, 5))
sns.barplot(y=[x[0] for x in data], x=[x[1] for x in data], palette='crest',orient='horizontal')
plt.ylabel('Class Labels')
plt.xlabel('Number of Images')
plt.title('Distribution of Class Labels');

# ### As we can observe from the plot, the data is much more balanced now compared to the original dataset. Also, we have doubled the size of the dataset

# **Hyperparamters**

# In[21]:


# ImageNet architectures excpect the image shapes to be 224 x 224
image_shape = (224,224,3)
batch_size = 64
epochs = 100
no_classes = 11

# ## Preparing the Data for the model

# - We've split the data into `80% for training` and `20% for testing`. 
# - Augmentation techniques have been applied to certain classes within the training dataset.
# - Since we've already augmented the dataset and saved the augmented images to disk, therefore we are now using an ImageDataGenerator to load the augmented images during training only.

# In[22]:


train_image_gen = ImageDataGenerator(rescale=1/255,            # rescale the image by normalzing it
                                    validation_split=0.2)      # split the data into 80% validation , 20% testing

test_image_gen = ImageDataGenerator(rescale=1/255)             # rescale the image by normalzing it
                                 

# In[23]:


# Defining training dataset
train_image_ds = train_image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               subset = 'training')

# In[24]:


# Defining validation dataset
val_image_ds = train_image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               subset = 'validation')

# In[25]:


# Defining testing dataset
test_image_ds = test_image_gen.flow_from_directory(test_path,
                                                  target_size=image_shape[:2],
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)  # Do not shuffle to keep the order consistent

# In[26]:


images, one_hot_labels = next(train_image_ds)
# get the 'one' which is corresponding to target label
labels = np.argmax(one_hot_labels, axis=1)

# In[27]:


images[0].max()

# In[28]:


images[0].min()

# ##  Displaying sample of Augmented Data

# In[29]:


# invert the dictionary using dict comprehension
label_names = {value: key for key, value in train_image_ds.class_indices.items()}

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(7,7))
ax = ax.flatten()

for i in range(16):
    ax[i].imshow(images[i])
    ax[i].set_title('{}'.format(label_names[labels[i]]))
    ax[i].axis('off')
plt.tight_layout()
plt.show()

# # Model Training

# ## Transfer Learning
# ### Why use pre-trained model for Image Classification tasks?
# 
# Transfer learning is useful for image classification tasks due to several reasons:
# 
# - **Feature Extraction**: as our dataset is not considered a large one, a pre-trained convolutional neural networks (CNNs), such as those trained on ImageNet, have learned to extract rich and complex features from images.By leveraging these pre-trained models as feature extractors, transfer learning allows us to benefit from the representations learned by the CNNs on a diverse dataset. This can significantly improve the performance of image classification models, especially when working with limited labeled data.
# 
# - **Domain Adaptation**: Image classification tasks often involve diverse datasets with variations in image quality, lighting conditions, viewpoints, and object appearances. Pre-trained models have learned to capture generic features that are transferable across different domains.By fine-tuning a pre-trained model on a target dataset, transfer learning helps adapt the learned representations to the specific characteristics of the target dataset, improving classification accuracy.
# 
# - **Faster Convergence Time**: Training deep CNNs from scratch on large-scale image datasets can be computationally intensive and time-consuming. Transfer learning allows us to initialize the model with pre-trained weights, which serve as a good starting point for optimization. This often leads to faster convergence during training and shorter training times compared to training from scratch.
# 
# - **Mitigate the risk of Overfitting**: Pre-trained models act as effective regularizers by providing strong priors on the model's parameters. Fine-tuning a pre-trained model on a target dataset helps prevent overfitting, especially when the target dataset is small. Transfer learning encourages the model to learn task-specific features while retaining the generalization capabilities acquired from the source dataset.

# # EfficientNet Architecture

# ## EfficientNet Overview
# 
# EfficientNet is a family of convolutional neural network (CNN) architectures designed to achieve excellent accuracy with significantly fewer parameters and computational resources compared to previous models. It was introduced by Mingxing Tan and Quoc V. Le in their paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," published in 2019.
# 
# The key innovation behind EfficientNet is a systematic approach to scaling CNNs, called **compound scaling**, which uniformly scales the network’s depth, width, and resolution with a set of fixed scaling coefficients. This method contrasts with previous practices where networks were often scaled up arbitrarily (e.g., making the network deeper or wider only).
# <br>
# ![EfficientNet Architecture](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*-ENqv4TI0JuyY6Nq8XQlAA.png)
# 

# ## Key Features of EfficientNet:
# 
# ### Compound Scaling:
# - **Depth scaling** refers to increasing the number of layers in the network.
# - **Width scaling** means expanding the number of channels in each layer.
# - **Resolution scaling** involves increasing the input image size.
# - EfficientNet uses a compound coefficient (denoted as φ) to scale these dimensions in a balanced way. The coefficient determines how much additional resources are available and scales the depth, width, and resolution accordingly.
# 
# 
# ### Baseline Architecture (EfficientNet-B0):
# - The base model, EfficientNet-B0, was developed using neural architecture search (NAS) to optimize the baseline architecture, focusing on accuracy and efficiency.
# - This base model is then scaled using the compound scaling method to create the other EfficientNet models (B1 to B7), each providing higher accuracy and complexity.
# 
# ### Use of Mobile Inverted Bottleneck Convolution (MBConv):
# - EfficientNets heavily utilize MBConv, an inverted bottleneck architecture initially introduced in MobileNetV2, which uses depthwise separable convolutions. This helps in reducing the model size and computational requirements while maintaining high accuracy.
# 
# ### Squeeze-and-Excitation Optimization:
# - Each MBConv block includes a squeeze-and-excitation layer, which helps the network to focus on the most informative features by re-weighting the channels of the convolutional features.

# ## Models in the EfficientNet Family:
# - Starting from EfficientNet-B0 (the baseline), the models go up to EfficientNet-B7. Each successive version (B1 to B7) scales up the dimensions according to the compound scaling rule, using larger φ values.
# - EfficientNet also extends into more efficient and specialized versions like EfficientNetV2, which introduces additional improvements and optimizations for specific tasks and hardware.

# ## EfficientNet B0 Architecture:
# ![EfficientNet B0 Architecture](https://www.researchgate.net/publication/348470984/figure/fig2/AS:979961129209859@1610652348348/The-EffecientNet-B0-general-architecture.png)

# **Steps:**
# 1. **Load Pre-trained Model:**
#    - Load the pre-trained EfficientNet-B0 model with pre-trained weights from the ImageNet dataset using the `EfficientNetB0` module in Keras. Exclude the fully connected layers at the top (`include_top=False`) to customize them for a specific task.
# 
# 2. **Add Custom Classification Head:**
#    - Add custom layers on top of the pre-trained model to adapt it for a specific classification task:
#      - Global Average Pooling layer to reduce spatial dimensions.
#      - Dense layer with ReLU activation to learn complex patterns.
#      - Dense output layer with softmax activation to produce class probabilities.
# 
# 3. **Create the Model:**
#    - Use the `Model` function to create a new model by specifying the input and output layers. The input layer is the input layer of the pre-trained model, and the output layer is the last layer added (predictions layer).

# In[30]:


# Load pre-trained EfficientNet-B5 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(11, activation='softmax')(x) 
EffB0_model = Model(inputs=base_model.input, outputs=predictions)

# In[31]:


early_stop = EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights=True )

# In[32]:


# Compile the model with SGD optimizer
sgd = SGD(0.01, momentum=0.9, nesterov=True)
EffB0_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# In[33]:


EffB0_model_history = EffB0_model.fit(train_image_ds,
                    epochs=epochs,               
                    validation_data=val_image_ds,
                    callbacks=[early_stop])

# # Model Evaluation

# In[34]:


# Loss and Accuracy on validation data
EffB0_model.evaluate(val_image_ds)

# In[35]:


# Loss and Accuracy on test data (new unseen data)
EffB0_model.evaluate(test_image_ds)

# In[36]:


# Generate predictions using the trained model on the test dataset.
EffB0_predictions_prob =EffB0_model.predict(test_image_ds)

# In[37]:


# Select the class with the highest probability as the predicted class
EffB0_predictions = np.argmax(EffB0_predictions_prob, axis=1)

# In[38]:


print(classification_report(test_image_ds.classes,EffB0_predictions))

# This classification report provides a comprehensive evaluation of a model's performance across 11 different classes. Here's a breakdown of the key metrics:
# 
# - **Precision:** Precision measures the accuracy of positive predictions made by the model. For instance:
#   - Class 0: The precision of 0.94 indicates that 94% of the instances predicted as class 0 are correct.
#   - Class 5: Achieving a precision of 1.00 suggests that all predictions for class 5 are correct.
# 
# - **Recall:** Recall, also known as sensitivity, gauges the model's ability to correctly identify all relevant instances. For instance:
#   - Class 3: The recall of 0.71 indicates that 71% of instances belonging to class 3 were correctly identified by the model.
#   - Class 7: With a recall of 1.00, the model accurately detects all instances from class 7.
# 
# - **F1-score:** The F1-score, the harmonic mean of precision and recall, offers a balance between these two metrics. For example:
#   - Class 2: Achieving an F1-score of 0.87 reflects a balanced performance between precision (0.82) and recall (0.92).
#   - Class 6: With an F1-score of 0.89, the model maintains a good balance between precision (0.95) and recall (0.82).
# 
# - **Support:** Support represents the number of actual occurrences of each class in the dataset. For instance:
#   - Class 8: With a support of 229, there are 229 instances of class 8 in the dataset.
# 
# - **Accuracy:** The overall accuracy of the model across all classes is 91%. This suggests that the model correctly predicts the class labels for 91% of the instances in the dataset.
# 
# - **Macro Average:** The macro average computes the average of precision, recall, and F1-score for all classes, treating each class equally. In this report, the macro average for precision, recall, and F1-score is approximately 0.92, indicating balanced performance across classes.
# 
# - **Weighted Average:** The weighted average calculates the metrics while considering the imbalance in class distribution by weighting each class's score by its support. Here, the weighted average for precision, recall, and F1-score is approximately 0.92, indicating consistent performance across classes, considering their respective support.
# 
# Overall, the classification report provides valuable insights into the model's performance, highlighting its strengths and areas for improvement across different classes.
# 

# In[40]:


plt.figure(figsize=(10,8))
cm = confusion_matrix(test_image_ds.classes,EffB0_predictions)
ax = sns.heatmap(cm,annot=True,fmt='d',cmap='mako')
ax.set_xticklabels(test_image_ds.class_indices.keys())
ax.set_yticklabels(test_image_ds.class_indices.keys())

plt.xlabel('True Labels')
plt.ylabel('Predicted Values')
plt.title('Confusion Matrix');

# # Result Visualization

# In[41]:


metrics = pd.DataFrame(EffB0_model_history.history)

# In[42]:


plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
plt.plot(metrics['accuracy'], label='Training Accuracy')
plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Accuracy vs. Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss vs. Epochs")
plt.legend()
plt.tight_layout()
plt.suptitle('EfficientNetB0 Based Model Peformance', y = 1.05)
plt.show()

# In[43]:


#Save the model
EffB0_model.save('EfficientNetB0_model.h5')

# ## EfficientNetB4 Architecture 
# 
# ![EfficientNet-B4 Model Architecture](https://www.researchgate.net/publication/350928633/figure/fig2/AS:11431281187544980@1694280215401/Our-modified-19-EFFICIENTNET-b4-architecture-Data-flow-is-from-left-to-right-a.tif)
# 
# 

# **Steps:**
# 1. **Load Pre-trained Model:**
#    - Load the pre-trained EfficientNet-B4 model with pre-trained weights from the ImageNet dataset using the `EfficientNetB4` module in Keras. Exclude the fully connected layers at the top (`include_top=False`) to customize them for a specific task.
# 
# 2. **Add Custom Classification Head:**
#    - Add custom layers on top of the pre-trained model to adapt it for a specific classification task:
#      - Global Average Pooling layer to reduce spatial dimensions.
#      - Dense layer with ReLU activation to learn complex patterns.
#      - Dense output layer with softmax activation to produce class probabilities.
# 
# 3. **Create the Model:**
#    - Use the `Model` function to create a new model by specifying the input and output layers. The input layer is the input layer of the pre-trained model, and the output layer is the last layer added (predictions layer).

# In[82]:


# Load pre-trained EfficientNet-B4 model
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
predictions = Dense(11, activation='softmax')(x) 
EffB4_model = Model(inputs=base_model.input, outputs=predictions)

# In[83]:


early_stop = EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights=True )

# In[84]:


# Compile the model with SGD optimizer
sgd = SGD(0.01, momentum=0.9, nesterov=True)
EffB4_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# In[85]:


EffB4_model_history = EffB4_model.fit(train_image_ds,
                    epochs=epochs,
                    validation_data=val_image_ds,
                    callbacks=[early_stop])

# # Model Evaluation

# In[86]:


# Loss and Accuracy on validation data
EffB4_model.evaluate(val_image_ds)

# In[87]:


# Loss and Accuracy on test data (new unseen data)
EffB4_model.evaluate(test_image_ds)

# In[88]:


# Generate predictions using the trained model on the test dataset.
predictions_prob =EffB4_model.predict(test_image_ds)

# In[62]:


# Select the class with the highest probability as the predicted class
EffB4_model_predictions = np.argmax(predictions_prob, axis=1)

# In[64]:


print(classification_report(test_image_ds.classes,EffB4_model_predictions))

# This classification report provides a detailed evaluation of a model's performance across 11 different classes. Here's an interpretation of the key metrics:
# 
# - **Precision:** Measures the accuracy of positive predictions made by the model. For instance:
#   - Class 1 has a precision of 0.88, indicating that 88% of the instances predicted as class 1 are correct.
# 
# - **Recall:** Reflects the model's ability to correctly identify all relevant instances. For example:
#   - Class 5 achieves perfect recall (1.00), indicating that all instances of class 5 were correctly identified.
# 
# - **F1-score:** Represents the harmonic mean of precision and recall, providing a balance between the two metrics. For instance:
#   - Class 4 has an F1-score of 0.95, reflecting a balanced performance between precision (0.98) and recall (0.93).
# 
# - **Support:** Denotes the number of actual occurrences of each class in the dataset. For example:
#   - Class 8 has a support of 229, indicating there are 229 instances of class 8 in the dataset.
# 
# - **Accuracy:** The overall accuracy of the model across all classes is 90%, indicating that the model correctly predicts the class labels for 90% of the instances in the dataset.
# 
# - **Macro Avg:** Computes the average of precision, recall, and F1-score for all classes, treating each class equally. In this report, the macro average for precision, recall, and F1-score is approximately 0.91, indicating balanced performance across classes.
# 
# - **Weighted Avg:** Calculates the metrics while considering the imbalance in class distribution by weighting each class's score by its support. Here, the weighted average for precision, recall, and F1-score is approximately 0.90, indicating consistent performance across classes, considering their respective support.
# 
# Overall, the classification report provides valuable insights into the model's performance, highlighting its strengths and areas for improvement across different classes.
# 

# In[65]:


plt.figure(figsize=(10,8))
cm = confusion_matrix(test_image_ds.classes,predictions)
ax = sns.heatmap(cm,annot=True,fmt='d',cmap='mako')
ax.set_xticklabels(test_image_ds.class_indices.keys())
ax.set_yticklabels(test_image_ds.class_indices.keys())

plt.xlabel('True Labels')
plt.ylabel('Predicted Values')
plt.title('Confusion Matrix');

# # Result Visualization

# In[71]:


metrics = pd.DataFrame(EffB4_model_history.history)

# In[72]:


plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
plt.plot(metrics['accuracy'], label='Training Accuracy')
plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Accuracy vs. Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss vs. Epochs")
plt.legend()
plt.tight_layout()
plt.suptitle('EfficientNetB7 Based Model Peformance', y = 1.05)
plt.show()

# In[73]:


#Save the model
EffB4_model.save('EfficientNetB7-201.h5')

# In[ ]:



