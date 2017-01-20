import tensorflow as tf
import os
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from tensorflow.contrib import losses
from tensorflow.examples.tutorials.mnist import input_data

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("//home//tensorflow//")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels

X_test = mnist.test.images
y_test = mnist.test.labels

X_validation = mnist.validation.images
y_validation = mnist.validation.labels

# Creating custom estimator
def model_function(features, targets, mode):
     # Configure the single layer perceptron model
     outputs = layers.fully_connected(inputs= features,
                                      num_outputs=10,
                                      activation_fn=None)
     
      # Calculate loss using mean squared error
     loss = losses.sigmoid_cross_entropy(outputs, targets)
      
      # Create an optimizer for minimizing the loss function
     optimizer = layers.optimize_loss(loss=loss,
                                       global_step= tf.contrib.framework.get_global_step(),
                                       learning_rate = 0.001,
                                       optimizer = "SGD")
     probs = tf.sigmoid(outputs)
     return {'probs': probs, 'labels':tf.arg_max(probs,1)}, loss, optimizer
              
#create custom estimator
nn = learn.Estimator(model_fn= model_function, model_dir="//home//tensorflow//Models//CustomEstimatormodel//Model3")

#Build the model
nn.fit(x=X_train, y =y_train, steps=1000, batch_size=1)
for var in nn.get_variable_names():
    print "%s:%s" %(var, nn.get_variable_value(var))
    
# Predict the outcome of test data using model
predictions = nn.predict(X_test,as_iterable=True)
for i, p in enumerate(predictions):
    print("Predictions %s: %s :%s " %(i+1, p['probs'], p['labels']))
