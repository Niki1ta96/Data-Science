# -*- coding: utf-8 -*-/
"""
Created on Tue Sep  5 15:59:37 2017

@author: nikita.jaikar
"""

import os
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

#os.chdir("E:/")
os.chdir('C:/Users/nikita.jaikar/Documents/Deep Learning/Keras/')
    
movie_train = pd.read_csv("Dataset/labeledTrainData.tsv", header=0, 
                    delimiter="\t", quoting=3)
movie_train.shape
movie_train.info()
movie_train.loc[0:4,'review']

#text cleaning example for one sample review
review_tmp = movie_train['review'][0]
review_tmp = BeautifulSoup(review_tmp).get_text()
review_tmp = re.sub("[^a-zA-Z]"," ", review_tmp)
review_tmp_words = review_tmp.split(' ')

def preprocess_review(review):        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case
        return review_text.lower()

def tokenize(review):
    return review.split()

vectorizer = text.CountVectorizer(preprocessor = preprocess_review, 
                                  tokenizer = tokenize,    
                                  stop_words = 'english',   
                                  max_features = 5000)

#transform the reviews to count vectors(dtm)
features = vectorizer.fit_transform(movie_train.loc[:,'review']).toarray()

#returns the stopwords used by count vectorizer
vectorizer.get_stop_words()
#get the mapping between the term features and dtm column index
vectorizer.vocabulary_
#get the feature names
vocab = vectorizer.get_feature_names()

#check the distribution of features across reviews
dist = np.sum(features, axis=0)
for tag, count in zip(vocab, dist):
    print(count, tag)