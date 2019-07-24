#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Using transfer learning for feature extraction
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


# In[3]:


import os
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[4]:


# base_model = VGG16(weights='imagenet', include_top=False)
# base_model.summary()


# In[5]:


# Storing image directory in a variable
img_path = "E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Labelled Images\\00DjNzR.png"
img_dir = "E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Labelled Images"


# In[6]:


# Reading data from csv
final_DF = pd.read_csv("E:\MSc DA\Sem 2\Project\Code and Docmentation\Final Dataset\sent_process_DF_2.csv", sep = '\t')
# Getting rid of unnecessary columns
final_DF = final_DF.iloc[:,1:]


# In[7]:


# taking in array of sentences
sentences=final_DF['sentence'].values
# taking in array of labels
t_y=final_DF['label'].values
# Instance of encoder
Encoder = LabelEncoder()
# encoding labels to binary (numerical) values 
y = Encoder.fit_transform(t_y)


# In[8]:


# replacing labels with 1's and 0's
final_DF['label'] = y


# In[9]:


# Test
final_DF.head()


# In[10]:


# storing image directories as a list
all_imgs = []
for root,j,files in os.walk(img_dir):
    for file in files:
        file = root+ '\\' + file
        all_imgs.append(file)


# In[11]:


# Testing
all_imgs[7]


# In[12]:


# Summary of the steps down below:
# 1. Get input : input_path -> image
# 2. Get output : input_path -> label
# 3. Pre-process input : image -> pre-processing step -> image
# 4. Get generator output : (batch_input, batch_labels )


# In[13]:


# Function that returns image reading from the path
def get_input(path):
    # Loading image from given path    
    img = image.load_img(path, target_size=(224,224))    
    return(img)


# In[14]:


# Testing
get_input("E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Labelled Images\\0Jzts4J.png")


# In[15]:


# Testing
fname = ("E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Labelled Images\\00DjNzR.png").split('\\')[-1]
(final_DF[final_DF['image_name'] == fname]['label']).shape[0]


# In[16]:


# Function to get the output
# returns an array of labels
def get_output(path,label_file=None):
    # Spliting the path and take out the image id    
    filename = path.split('\\')[-1]
    # Taking list of labels
    labels = list(label_file[label_file['image_name'] == filename]['label'].values)
    # for duplicate selecting labels
    if len(labels) <= 2:
        label = labels[0]
    elif len(labels) > 2:
        uni_label = list(set(labels))
        count_label = [labels.count(lab) for lab in uni_label]
        lab_idx = count_label.index(max(count_label))
        label = uni_label[lab_idx]
    return label


# In[17]:


# Testing 
get_output("E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Labelled Images\\0UQh5Eo.png", final_DF)


# In[18]:


# Takes in image and preprocess it
def process_input(img):
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return(img_data)


# In[19]:


# Testing
meme = get_input("E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Labelled Images\\0P5i3yI.png")
process_input(meme).shape


# In[20]:


# Function to generate the data
def image_generator(files,label_file, batch_size = 64):    
    while True:        
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a = files, 
                                         size = batch_size)
        batch_input = []
        batch_output = [] 
          
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            input = get_input(input_path )
            output = get_output(input_path,label_file=label_file )
#             print(output)
            input = process_input(img=input)
            batch_input.append(input[0]) 
            batch_output.append(output) 
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        
        yield( batch_x, batch_y )


# In[21]:


# Testing
image_generator(all_imgs, final_DF, batch_size=3)


# In[22]:


# create base model with imagenet weights
base_model = VGG16(weights='imagenet', include_top=False)
base_model.summary()


# In[23]:


# Freezing all the trainable layers
for layer in base_model.layers:
    layer.trainable = False


# In[24]:


base_model.summary()


# In[25]:


# Creating output layer
x = base_model.output
# Adding pooling layer before the output
x = GlobalAveragePooling2D()(x)
# Adding a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer with 2 classes
predictions = Dense(1, activation='softmax')(x)


# In[26]:


# Defining model
model = Model(inputs=base_model.input, outputs=predictions)


# In[27]:


model.summary()


# In[28]:


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[29]:


# Creating generator object and storing it in a variable
img_gen = image_generator(all_imgs, final_DF, batch_size=16)


# In[30]:


final_DF.head()


# In[31]:


# Training model
model.fit_generator(img_gen, epochs=1, steps_per_epoch=5)


# In[32]:


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from VGG16. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)


# In[33]:


# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True


# In[34]:


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics = ['accuracy'])


# In[35]:


# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(img_gen, epochs=1, steps_per_epoch=5)

