#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Importing necessary libraries
import urllib, re 
#Selenium libraries
from selenium import webdriver
import os
import urllib.request


# In[125]:


#Defining function to check if image is present on url
def url_status(url):
    try:
        r = urllib.request.urlopen(url)
        return r.code
    except urllib.request.HTTPError as e:
        r = e
        return r.code
    except urllib.request.URLError as e:
        r=e
        return e.args


# In[99]:


# Testing
# Uncomment only if needs to be tested
# url_status('http://imgur.com/a/IazT5')


# In[3]:


# Function to get image urls from imgur 
def scrap_imgur(url, filename):
    url += '.png'
    filename += '.png'
    work_dir = os.getcwd() + '\\Meme Data\\imgur\\'
#     urllib.request.urlretrieve(url, work_dir + filename)
    return url


# In[4]:


# Testing
# Uncomment if needs to be tested
# scrap_imgur('http://imgur.com/LgPKrP1', 'filename')


# In[5]:


# Importing headless selenium driver to deal FB images
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import pandas as pd


# In[6]:


# Headless drivers
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920x1080")
driver = webdriver.Chrome(options=chrome_options)


# In[7]:


#Function to scrap FB images
def scrap_fb(url, filename):
    driver.get(url)
    images = driver.find_elements_by_class_name("scaledImageFitWidth")
    # Importing image url from page
    filename += '.png'
    work_dir = os.getcwd() + '\\Meme Data\\FB\\'
    try:
        url = images[1].get_attribute('data-src')
#         urllib.request.urlretrieve(url, work_dir + filename)
    # Avoiding index error      
    except IndexError:
        url = 'null'
#     urllib.request.urlretrieve(url, work_dir + filename)
    return url


# In[8]:


# Testing
# Uncomment if needs to tested
# scrap_fb('https://facebook.com/1667496210185013/posts/1757866207814679', 'filename')


# In[9]:


# Importing beautiful soup to take out twitter data
from bs4 import BeautifulSoup as Soup
from urllib.request import urlopen as ure


# In[10]:


# Function to scrap images from twitter url
def scrap_twitter(url, filename):
    soup = ure(url)
    temp_soup = Soup(soup)
    try:
        url = temp_soup.find_all('div', {'class':"AdaptiveMedia-photoContainer"})[0].find('img').get('src')
    except IndexError:
        url = 'null'
    #Importing image url from page
    filename += '.png'
    work_dir = os.getcwd() + '\\Meme Data\\Twitter\\'
#     urllib.request.urlretrieve(url, work_dir + filename)
    return url


# In[11]:


# Importing random for random sampling 
import random
# Importing numpy to carry out array operations
import numpy as np


# In[12]:


# Importing re for regex
import re


# In[13]:


# Removing longer captions from the data
def count_lines(line):
    split_sent = line.split('\\n')
    match_words = [re.match(' [0-9]*:[0-9]* PM',i) for i in split_sent]
    boo_list = [i is None for i in match_words]
    PreList = [a*b for a,b in zip(split_sent,boo_list)]
    PreList = [i for i in PreList if i != '']
    return len(PreList)


# In[15]:


# Function to process dataframe
def preprocess_data(dirct):
    # df_data = pd.read_csv(dirct, engine = 'python', sep = '\t')
    df_data = pd.read_csv(dirct, engine = 'python')
    # Selecting required columns
    df_data = df_data[['id','link','caption', 'network']]
    # Removing data related to instagram 
    df_data = df_data[df_data['network'] != 'instagram']
    # Creating empty column named status to keep check on the status of the urls i.e. modified
    df_data['status'] = ""
    # Creating cap_len column to store number of lines in an observation    
    df_data['cap_len'] = [count_lines(sent) for sent in df_data['caption']]
    # Subsetting dataframe as per cap_len 
    df_data = df_data[df_data['cap_len'] < 20]
    # Taking out all the id
    id_list = df_data['id']
    # Making list of ids
    list_id = [ID for ID in id_list]
    random.seed(99)
    if len(id_list) > 500:        
        # Taking out 500 random ids
        rand_ids = random.sample(list_id, 500)
    else:
        rand_ids = list_id
    # Creating new data frame according to ids
    df_new = pd.DataFrame()
    for i in range(len(rand_ids)):
        df_new = df_new.append(df_data[df_data['id'] == rand_ids[i]])
    # Checking url 
    true_urls = [(df_new.index[i], link) for i,link in enumerate(df_new['link']) if url_status(link) in (200, 401)]  
    # Empty DataFrame to append the rows with working urls    
    df_final = pd.DataFrame()
    # Comparing the url index of working urls with preprocessed dataframe earlier     
    for i in range(len(true_urls)):
        df_final = df_final.append(df_new[df_new.index == true_urls[i][0]])
    return df_final


# In[17]:


# Testing 
# Uncomment if needs to be tested
# test_df = preprocess_data("E:\\MSc DA\\Sem 2\\Project\\2016electionmemes\\Feel_the_Bern.csv")


# In[21]:


# Storing base directory in variable
base_dir = 'E:\\MSc DA\\Sem 2\\Project\\2016electionmemes'


# In[22]:


# Creating list of csv files 
for root, dirs, files in os.walk(base_dir):
    paths_dir = [root + '\\' + name for name in files if name.endswith((".csv"))]


# In[24]:


# Testing
# Uncomment if needs to be tested
# paths_dir


# In[26]:


# Function to create url collection
def url_collection(DF):
    # As index is not aligned     
    df_idx = DF.index
    # Looping over the DataFrame to replace the existing urls with image urls    
    for i in range(len(DF)):
        # Reading out ID as file name which later could be used to save the image with same name         
        filename = str(DF.iloc[[i]]['id'][df_idx[i]])
        # Storing url in 'url' to provide it to function that extracts image url from the url provided         
        url = DF.iloc[[i]]['link'][df_idx[i]]
        status = DF['status'][df_idx[i]]
        # Putting if condition to avoid rewriting the url          
        # if (not(url.endswith('.png'))):
        if status != 'modified':
            # Replacing the 'imgur' url with image url            
            if (DF.iloc[[i]]['network'][df_idx[i]] == 'imgur'): 
                DF.at[df_idx[i], 'link'] = scrap_imgur(url, filename)
                DF['status'][df_idx[i]] = 'modified'
            # Replacing the 'Facebook' url with image url
            elif (DF.iloc[[i]]['network'][df_idx[i]] == 'facebook'):
                DF.at[df_idx[i], 'link'] = scrap_fb(url, filename)
                DF['status'][df_idx[i]] = 'modified'
            # Replacing the 'Twitter' url with image url
            else:
                DF.at[df_idx[i], 'link'] = scrap_twitter(url, filename)
                DF['status'][df_idx[i]] = 'modified'
        else:
            DF.at[df_idx[i], 'link'] = url
    return DF


# In[27]:


# Storing images on local and creating working csv with image urls
def creating_op(dirct):
    # Processing Dataframe    
    processed_df = preprocess_data(dirct)
    # Collecting true urls
    refined_df = url_collection(processed_df)  
    # Removing the null urls
    refined_df = refined_df[refined_df['link'] != 'null']
    # Defining ouptup directory     
    op_dir = dirct.replace('2016electionmemes\\', '2016electionmemes\\Refined\\')
    # Writing file at above location    
    refined_df.to_csv(op_dir, sep='\t', encoding='utf-8')


# In[202]:


# Uncomment only if needs to be run
# for path in paths_dir:
#     creating_op(path)


# In[63]:


# Creating output directory
base_dir = "E:\\MSc DA\\Sem 2\\Project\\2016electionmemes\\Refined"
for root, dirs, files in os.walk(base_dir):
    op_dir = [root + '\\' + name for name in files if name.endswith((".csv"))]


# In[64]:


count_per_file_op = [len(pd.read_csv(j, engine = 'python', sep='\t')) for j in op_dir]
# len(pd.read_csv(op_dir[0], engine='python', sep='\t'))


# In[203]:


# Printing the max number of tokens
# sum(count_per_file_op)


# In[67]:


# Taking in all the dataframes in single list
combined_DF = [pd.read_csv(i, sep= '\t', encoding = 'utf-8') for i in op_dir]


# In[204]:


# Checking if all the files are part of the list
# len(combined_DF)


# In[69]:


# appending all the lengths 
Caption_len = []
for DF in combined_DF:
    Caption_len.append([count_lines(i) for i in DF['caption']])        


# In[72]:


# taking guess of lengths in the data
New_cap_len = [i for i in Caption_len if i != []]
# Maximum caption lengths in each file
[max(cap_len) for cap_len in New_cap_len]


# In[73]:


# concatenating all the dataframes
con_Df = pd.concat(combined_DF)
# con_Df['cap_len'] = [count_lines(i) for i in con_Df['caption']]


# In[205]:


# Number of rows in new Dataframe 
# Keeping caption lenght lower than 20
len(con_Df[con_Df['cap_len'] < 20])


# In[126]:


# Adding new column URL_status to the existing dataframe to store status of URL (200,404, 401)
con_Df['URL_status'] = [url_status(con_Df.iloc[i]['link']) for i in range(len(con_Df['link']))]


# In[165]:


# url_status(con_Df['link'][0])
con_Df = con_Df[con_Df['URL_status'] != 404]


# In[167]:


# Writing csv output
con_Df.to_csv("E:\MSc DA\\Sem 2\\Project\\2016electionmemes\\Refined\\Combined_Df.csv", sep='\t', encoding='utf-8')


# In[143]:


# Defining function to make list of dataframe
def chunks(DF, n):
    n = max(1, n)
    return (DF.iloc[i:i+n, :] for i in range(0, len(DF), n))


# In[157]:


# List of dataframe by using chunks() function
list_con_Df = [i for i in chunks(con_Df,30)]


# In[168]:


# Writing csvs in the form
for i in range(len(list_con_Df)):
    Filename = 'Memes_Data_Survey_'+ str(i) + '.csv'
    list_con_Df[i].to_csv('E:\\MSc DA\\Sem 2\\Project\\2016electionmemes\\Form csvs\\' + Filename, sep='\t', encoding='utf-8')


# In[182]:


# Creating directory of the forms
form_dir = 'E:\\MSc DA\\Sem 2\\Project\\2016electionmemes\\Form csvs\\Forms\\'


# In[208]:


# form_dir


# In[199]:


# Creating csv compatible for google forms
def comp_DF(DF_list):
    for i in range(len(DF_list)):
        with open(form_dir + 'Memes_Data_Survey_' + str(i) + '.csv','w',encoding='utf-8') as k:
            for index, row in DF_list[i].iterrows():
                if (index % 10 )==0:
                    print('IMAGE\t Choose the option\t',row.iloc[3],'\t',
                          row.iloc[2]+'\t','offensive\t Non-offensiv', file=k)
                    # Adding Page to limit the number of images on the page
                    print('PAGE',file=k)
                else:
                    print('IMAGE\t Choose the option\t',row.iloc[3],'\t',
                          row.iloc[2]+'\t','offensive\t Non-offensiv', file=k)


# In[201]:


# Using function created above
# Uncomment only if needs to executed
# comp_DF(list_con_Df)

