import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import pandas as pd
import numpy as np
import os
from sklearn import model_selection 

os.chdir("//home//tensorflow")

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

# Read the train data in pandas dataframe
sample = pd.read_csv("train1.csv")
sample.shape
sample.info()

X = learn.extract_pandas_data(sample[['x1','x2']])
y = learn.extract_pandas_labels(sample[['label']])

# Divide the input data into train and validation
X_train, X_validation,y_train,y_validation = model_selection.train_test_split(X,y, test_size = 0.2, random_state=196)
type(X_train)

#feature engineering
features_col = [layers.real_valued_column("",dimension=2)]
 
classifier = learn.LinearClassifier(feature_columns= features_col,
                                    n_classes= 2,
                                    model_dir="//home//tensorflow//Models//model3")
#build the model
classifier.fit(x=X_train, y=y_train, steps = 1000)
classifier.weights_
classifier.bias_

#evaluate the model using validation set
output = classifier.evaluate(x=X_validation, y = y_validation, steps = 1000)
type(output)
for key in sorted(output):
    print "%s:%s" %(key, output[key])
    
# Predict the outcome of test data using model
test = np.array([[100.4,21.5,10.5,22.4],[200.1,26.1,2.7,26.7]])
predictions = classifier.predict(test)
predictions


                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
