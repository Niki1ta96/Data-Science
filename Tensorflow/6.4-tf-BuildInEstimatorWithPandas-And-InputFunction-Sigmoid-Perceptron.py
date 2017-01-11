import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
import pandas as pd
import numpy as np
import os

os.chdir("//home//tensorflow")
# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

# Read the train data
sample = pd.read_csv("train1.csv")
sample.shape
sample.info()

FEATURES = ['x1','x2']
LABEL = ['label']

# input function must return featurecols and labels separately  
def input_function(dataset, train=False):
    feature_cols = {k : tf.constant(dataset[k].values) 
                        for k in FEATURES}
    if train:
        labels = tf.constant(dataset[LABEL].values)
        return feature_cols, labels
    return feature_cols
    
# Build the model with right feature tranformation
faetures_cols = [layers.real_valued_column(k) for k in FEATURES]

classifier = learn.LinearClassifier(feature_columns= faetures_cols,
                                    n_classes= 2,
                                    model_dir="//home//tensorflow//Models//model3")

classifier.fit(input_fn= lambda:input_function(test, False), steps = 1000)

# Predict the outcome using model
dict = {'x1':[10.4,21.5,10.5], 'x2':[22.1,26.1,2.7] }
test = pd.DataFrame.from_dict(dict)

predictions = classifier.predict(input_fn = lambda: input_function(test,False))
predictions

