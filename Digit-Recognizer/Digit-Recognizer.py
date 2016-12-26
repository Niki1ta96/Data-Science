import os
import pandas as pd
import numpy as np
import random as r # can also use numpy for random() function



os.chdir("C:\\Users\\nikita.jaikar\\Documents\\GitHub\\Data-Science\\Digit-Recognizer")
os.getcwd()

digit_train = pd.read_csv("C:\\Users\\nikita.jaikar\\Documents\\GitHub\\Data-Science\\Data\\digit_train.csv")
type(digit_train)#pandas.core.frame.DataFrame
digit_train.shape #Dimension
digit_train.info() # Summary
digit_train.dtypes # type of evry column

# Data Exploration 
digit_train.head()
digit_train.tail()
digit_train.head(3)

#Equivalent to Summary in R
digit_train.describe()

#Read test data 
digit_test = pd.read_csv("C:\\Users\\nikita.jaikar\\Documents\\GitHub\\Data-Science\\Data\\digit_test.csv")
digit_test.shape

# randomly predict the labels
digit_test.label = r.randrange(10) # [0,9]
#Vectorization NOT avaliabl;e by default on basic data types
# Vectorization only in numpy in np.array

digit_train1 = digit_train[[0,1]]

imageid = range(1, 28001,1)
len(imageid)

label = np.random.randint(0,9,28000)
dict = {'Imageid':imageid, 'Label':label}
out_df = pd.DataFrame.from_dict(dict)
out_df.shape
out_df.head()
out_df.set_index('Imageid', inplace = True)
out_df.to_csv("Submission.csv")
