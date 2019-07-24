#!/usr/bin/env python
# coding: utf-8

# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd


# In[50]:


from sklearn import model_selection, naive_bayes, svm


# In[84]:


# df = pd.read_csv('first_try.csv', names=['image_name','sentence', 'label'], sep='\t')
df = pd.read_csv("E:\\MSc DA\\Sem 2\\Project\\Code and Docmentation\\Final Dataset\\sent_process_DF.csv", sep='\t')
df = df.iloc[:,1:]


# In[85]:


df.head()


# In[90]:


# taking in array of sentences
sentences=df['sentence'].values
# taking in array of labels
t_y=df['label'].values
# Instance of encoder
Encoder = LabelEncoder()
# encoding labels to binary (numerical) values 
y = Encoder.fit_transform(t_y)


# In[91]:


# Testing
sentences[0]


# In[92]:


# Dividing dataset into test and train using sklearn library
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)


# In[93]:


# Testing
print(sentences_train, sentences_test, y_train, y_test)


# In[94]:


# instance of count vectoriser
vectorizer = CountVectorizer()
# converting sentences into vectors
vectorizer.fit(sentences_train)
# Creating training and test vectors
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)


# In[95]:


#Baseline Logistic Regression
# Instance of classifier
classifier = LogisticRegression()
# Training classifier 
classifier.fit(X_train, y_train)
# Performace Metric Accuracy 
score = classifier.score(X_test, y_test)
print("Accuracy:", score)


# In[96]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train,y_train)
# predict the labels on validation dataset
predictions_NB = Naive.predict(X_test)
# Use accuracy_score function to get the accuracy
score = Naive.score(X_test, y_test)
print("Naive Bayes Accuracy Score -> ",score)


# In[97]:


# classifier


# In[98]:


# Testing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
vocab_size = len(tokenizer.word_index) + 1
print(sentences_train[2])
print(X_train[2])


# In[99]:


# Assigning limits to vector length
maxlen = 100
# Padding X_train and X_test with 0 to match the length
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(X_train[0, :])


# In[100]:


# NN Model
# Creating word embeddings
embedding_dim = 50
# Initiating sequential model
model = Sequential()
# Adding embedding layer
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[101]:


# Training the given model 
history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[ ]:





# In[ ]:




