import os
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import input_data

os.chdir("/home//tensorflow//")
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets('MNIST_data', one_hot= True)

X_train = mnist.train.images
y_train = mnist.train.labels

X_validation = mnist.validation.images
y_validation = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels

# creating custom estimator
def model_function(features, targets, mode):
    # Configure the single layer perceptron model
    outputs = layers.fully_connected(inputs=features,
                                     num_outputs=10,
                                     activation_fn=None)
    # Calculate loss using mean squared error
    loss = losses.softmax_cross_entropy(outputs, targets)
    
    # Create an optimizer for minimizing the loss function
    optimizer = layers.optimize_loss(loss=loss,
                                     global_step=tf.contrib.framework.get_global_step(),
                                     learning_rate= 0.001,
                                     optimizer='SGD')
    probs = tf.nn.softmax(outputs)
    return {'probs': probs, 'labels':tf.arg_max(probs,1)}, loss, optimizer
            
#create custom estimator
nn = learn.Estimator(model_fn=model_function, model_dir="//home//tensorflow//Models//CustomEstimatormodel//Model4")

#build the model
nn.fit(x=X_train,y=y_train, steps=1000, batch_size=1)
for var in nn.get_variable_names():
    print "%s:%s" % (var, nn.get_variable_names(var))

# Predict the outcome of test data using model
predictions = nn.predict(x=X_test, as_iterable=True)
for i,p in enumerate(predictions):
    print("Predictions %s: %s: %s" %(predictions, p['probs'],p['labels']))
    