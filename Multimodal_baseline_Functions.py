#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
# from nltk import word_tokenize
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D, Embedding, LSTM, multiply
from keras.models import Model
from keras import preprocessing, Input
import os
import itertools
import numpy as np
from PIL import Image, ImageFile


# In[2]:


STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
EMAIL = re.compile('^([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})$')
# NUMBERS = re.compile(['0-9'])
STOPWORDS = set(stopwords.words('english'))


# In[3]:


Training_path = "E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Split Dataset\\Training_meme_dataset.csv"
Validation_path = "E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Split Dataset\\Validation_meme_dataset.csv"
Testing_path = "E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Split Dataset\\Testing_meme_dataset.csv"
img_dir = "E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\Labelled Images"


# In[4]:


# For vectors
maxlen = 1000


# In[5]:


def encode_label(DataFrame, Label_col):
    t_y = DataFrame[Label_col].values
    Encoder = LabelEncoder()
    y = Encoder.fit_transform(t_y)
    DataFrame[Label_col] = y
    
def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()
    text = EMAIL.sub('', text)
#     text = NUMBERS.sub('',text)
    text = REPLACE_BY_SPACE_RE.sub(' ',text)
    text = BAD_SYMBOLS_RE.sub('',text)    
    text = text.replace('x','')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    
    return text


# In[6]:


def create_img_array(img_dirct):
    all_imgs = []
    for root, j, files in os.walk(img_dirct):
        for file in files:
            file = root + '\\' + file
            all_imgs.append(file)
    return all_imgs

def create_img_path(DF, Col_name, img_dir):
    img_path = [img_dir + '\\' + name for name in DF[Col_name]]
    return img_path


# In[7]:


def preprocess_text(Training_path,Validation_path, Testing_path):
    # function to preprocess input
    training_DF = pd.read_csv(Training_path, sep = ',')
    validation_DF = pd.read_csv(Validation_path, sep = ',')
    testing_DF = pd.read_csv(Testing_path, sep = ',')

    # encoding all the labels 
    encode_label(testing_DF,'label')
    encode_label(training_DF, 'label')
    encode_label(validation_DF, 'label')

    clean_text(training_DF['sentence'][0])

    # Processing the text
    training_DF['sentence'] = training_DF['sentence'].apply(clean_text)
    testing_DF['sentence'] = testing_DF['sentence'].apply(clean_text)
    validation_DF['sentence'] = validation_DF['sentence'].apply(clean_text)

    return training_DF, testing_DF, validation_DF


# In[8]:


# Function that returns image reading from the path
def get_input(path):
    # Loading image from given path
    # and resizing it to 224*224*3 format
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = image.load_img(path, target_size=(224,224))    
    return(img)

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

# Takes in image and preprocess it
def process_input(img):
    # Converting image to array    
    img_data = image.img_to_array(img)
    # Adding one more dimension to array    
    img_data = np.expand_dims(img_data, axis=0)
    #     
    img_data = preprocess_input(img_data)
    return(img_data)


# In[134]:


# Function to generate the data
def image_generator(files,label_file, batch_size = None):   
    """
        files: list of image paths 
        label_file: labels of the observations
        batch_size: Number of observations to be selected at a time
        
        return: generator object of image data
    """
    idxs = list(range(len(files)))
    idx = 0
    while True: 
        batch_paths = files[idx:idx+batch_size]
#         batch_paths = np.random.choice(a = files, size = batch_size)
        batch_input = [] # Batch input initialization
        batch_output = [] # Batch output initialization
          
        # Read in each input, perform preprocessing and get labels    
        for input_path in batch_paths:
            input = get_input(input_path ) # Load image
            output = get_output(input_path,label_file=label_file ) # Load label of the image
            input = process_input(img=input) # Process the image
            batch_input.append(input[0]) # Append the image
            batch_output.append(output)  # Append the label
            
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        if len(batch_x) < batch_size:
            idx = 0
        else:             
            yield (batch_x, batch_y)


# In[10]:


def text_generator(padded_seq, y, batch_size=None):
    """
        padded_seq: vectorized padded text sequence 
        y: label of the text
        batch_size: Number of observations to be selected at a time
        
        return: generator object of text data
    """
    idxs = list(range(len(y)))
    idx = 0
    while True:
        batch_idxs = idxs[idx:idx+batch_size]
        idx = idx + batch_size
#         batch_idxs = np.random.choice(a = list(range(len(padded_seq))), size=batch_size) #Selecting the random batch indexes    
        batch_input = [] # Initializing batch input
        batch_output = [] # Initializing batch output
        
        # Traversing through the batch indexes
        for batch_idx in batch_idxs:
            input = padded_seq[batch_idx] # selecting padded sequences from the batch
            output = y[batch_idx] # Selecting label            
            batch_input.append(input) # Appending the input (text vector)
            batch_output.append(output) # Appending the label
        
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        if len(batch_x) < batch_size:
            idx = 0
        else:             
            yield (batch_x, batch_y)


# In[147]:


#def img_text_generator(files, padded_seq, y, batch_size=None):
#    """
#        padded_seq: vectorized padded text sequence 
#        y: label of the text
#        batch_size: Number of observations to be selected at a time
#        
#        return: generator object of text data
#    """
#    idxs = list(range(len(padded_seq)))
#    idx = 0
#    while True:
#        batch_idxs = idxs[idx:idx+batch_size]
##         batch_idxs = np.random.choice(a = list(range(len(padded_seq))), size=batch_size) #Selecting the random batch indexes    
#        batch_input_txt = [] # Initializing batch input text
#        batch_input_img = [] # Initializing batch input image
#        batch_output = [] # Initializing batch output
#        
#        # Traversing through the batch indexes
#        for batch_idx in batch_idxs:
#            input_txt = padded_seq[batch_idx] # selecting padded sequences from the batch
#            output = y[batch_idx] # Selecting label  
#            input_img = get_input(files[batch_idx])
#            input_img = process_input(input_img)
#            batch_input_txt.append(input_txt) # Appending the input (text vector)
#            batch_input_img.append(input_img[0])
#            batch_output.append(output) # Appending the label
#        
#        # Return a tuple of (input,output) to feed the network
#        batch_x1 = np.array( batch_input_img )
#        batch_x2 = np.array( batch_input_txt )
#        batch_y = np.array( batch_output )
#        if (len(batch_x1) < batch_size):
#            idx = 0
#        else:             
#            yield ([batch_x1, batch_x2], batch_y)


#def image_generator(files,label_file, batch_size = None):    
#    while True:        
#        # Select files (paths/indices) for the batch
#        batch_paths = np.random.choice(a = files, 
#                                         size = batch_size)
#        batch_input = []
#        batch_output = [] 
#          
#        # Read in each input, perform preprocessing and get labels
#        for input_path in batch_paths:
#            input = get_input(input_path )
#            output = get_output(input_path,label_file=label_file )
##             print(output)
#            input = process_input(img=input)
#            batch_input.append(input[0]) 
#            batch_output.append(output) 
#        # Return a tuple of (input,output) to feed the network
#        batch_x = np.array( batch_input )
#        batch_y = np.array( batch_output )
#        
#        yield( batch_x, batch_y )

def img_text_generator(files, padded_seq, y, batch_size=None):
    """
        padded_seq: vectorized padded text sequence 
        y: label of the text
        batch_size: Number of observations to be selected at a time
        
        return: generator object of text data
    """
    while True:
        batch_idxs = np.random.choice(a = list(range(len(padded_seq))), size=batch_size) #Selecting the random batch indexes    
        batch_input_txt = [] # Initializing batch input text
        batch_input_img = [] # Initializing batch input image
        batch_output = [] # Initializing batch output
        
        # Traversing through the batch indexes
        for batch_idx in batch_idxs:
            input_txt = padded_seq[batch_idx] # selecting padded sequences from the batch
            output = y[batch_idx] # Selecting label  
            input_img = get_input(files[batch_idx])
            input_img = process_input(input_img)
            batch_input_txt.append(input_txt) # Appending the input (text vector)
            batch_input_img.append(input_img[0])
            batch_output.append(output) # Appending the label
        
        # Return a tuple of (input,output) to feed the network
        batch_x1 = np.array( batch_input_img )
        batch_x2 = np.array( batch_input_txt )
        batch_y = np.array( batch_output )
        yield ([batch_x1, batch_x2], batch_y)
