from tensorflow.contrib import layers
import pandas as pd
import tensorflow as tf

dict = {'id':range(1,6), 'pclass':[1,2,1,2,1], 'gender':['M','F','F','M','M'], 'fare':[10.5,22.3,11.6,22.4,31.5] }

df = pd.DataFrame.from_dict(dict)
df.shape
df.info()

# continuous types - real valued column
id = layers.real_valued_column('id')
type(id)
id.key

#real value column
fare = layers.real_valued_column('fare')
type(fare)
fare.key

cont_features = ['id','fare']

# comprehension for creating all real valued columns to do above work efficiently
cont_features_cols = [layers.real_valued_column(k) for k in cont_features]
                      
#bucketized column
# converting a continuous attribute to categorical/bucketized features
fare_bucket = layers.bucketized_column(fare, boundaries=[15,30])
type(fare_bucket)
fare_bucket.key

#converting continuous valued feature data to constant tensor
df['id']        ## equivalent to type(df.id)
df[['id']]
type(df['id'])      #pandas.core.series.Series
type(df[['id']])    #pandas.core.frame.DataFrame
df[['id']].size     #5
type(df[['id']].values)     #numpy.ndarray
ct = tf.constant(df[['id']].values)
type(ct)    #tensorflow.python.framework.ops.Tensor

# no [[]] here because we are using an element of a list (k)
# which itself is like list[]
cont_features = {k: tf.constant(df[k].values) for k in cont_features}
# data is in dictionary format 
                 
















