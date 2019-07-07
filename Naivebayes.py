#Importing the libraries
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

#setting random seed to 500
np.random.seed(500)

#reading dummy file in csv format
file_text=pd.read_csv("../data/shardul_data/text.csv", header=None, sep='\t')

#Creating corpus and seperating train and test 
train_x=file_text[0][:-4]
train_y=file_text[1][:-4]
corpus=file_text[0][:]
test_x=file_text[0][8:]
test_y=file_text[1][8:]

#to do find better way to convert the dataframe objects to list of sentences
#converting into list of sentences
train_x=[m for m in train_x]
train_y=[m for m in train_y]
test_x=[m for m in test_x]
test_y=[m for m in test_y]

#proper naming
#one-hot encoding
vectorizer = CountVectorizer()
vectorizer.fit(corpus)
X = vectorizer.transform(train_x)
print(X.toarray().shape)
print(X.toarray())
print(vectorizer.get_feature_names())
x = vectorizer.transform(test_x)
print(x.toarray().shape)

#encoding y-valuse
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(train_y)
Test_Y = Encoder.fit_transform(test_y)

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(X,Train_Y)


# predict the labels on validation dataset
predictions_NB = Naive.predict(x)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)