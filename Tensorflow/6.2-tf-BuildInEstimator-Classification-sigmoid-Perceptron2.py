import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import numpy as np
import pandas as pd
import os
from sklearn import model_selection


# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("//home//tensorflow")

#load the dataset
# reading directly using tensorflow's api
# train.csv does not have headers. Instead, first row has #rows, #columns
sample = learn.datasets.base.load_csv_with_header(
                    filename="train.csv",
                    target_dtype= np.int,
                    features_dtype=np.float,
                    target_column=-1)
X = sample.data
y = sample.target

# Divide the input data into train and validation
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(
                    X, y , test_size = 0.2, random_state = 196)
type(X_train)

#feature engineering
#feature_columns argument in classifier expects list of tensorflow feature types
features_cols = [layers.real_valued_column("", dimension=2)]
                 
#build the model configuration  
classifier = learn.LinearClassifier(
                    feature_columns=features_cols,
                    n_classes=2,
                    model_dir="//home//tensorflow//Models//model3")

#build the model
classifier.fit(x = X_train, y=y_train, steps = 1000)

#access the learned model parameters
classifier.weights_
classifier.bias_

#evaluate the model using validation set
results = classifier.evaluate(x=X_validation, y = y_validation, steps = 1000)
type(results)       # Dict

# to fetch values form dict result
for key in sorted(results):
    print "%s:%s" %(key, results[key])
    
# Predict the outcome of test data using model
test = np.array([[100.4,21.5,10.5,22.4],[200.1,26.1,2.7,26.7]])
predictions = classifier.predict(test)
predictions























