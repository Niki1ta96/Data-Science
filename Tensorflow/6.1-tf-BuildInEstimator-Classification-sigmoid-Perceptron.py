import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import pandas as pd
import numpy as np
import os

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("//home//tensorflow")
os.getcwd()

# reading directly using tensorflow's api
# train2.csv does not have headers. Instead, first row has #rows, #columns
sample = learn.datasets.base.load_csv_with_header(
                    filename="train.csv",
                    target_dtype=np.int,
                    features_dtype=np.float,
                    target_column=-1)

type(sample)
sample.data.shape
type(sample.data)
sample.target.shape
sample.target
type(sample.target)

type(sample.data[1])

#feature_columns argument expects list of tesnorflow feature types
features_col = [layers.real_valued_column(""), dimension=2]


classifier = learn.LinearClassifier(
                    feature_columns= features_col],
                    n_classes= 2,
                    model_dir="//home//tensorflow//Models//model1")

classifier.fit(x=sample.data, y= sample.target, steps = 1000)

classifier.weights_
classifier.bias_

















