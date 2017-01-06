import tensorflow as tf
import numpy as np

session = tf.InteractiveSession()

#Rank 0 tensor(Scalar)
c1 = tf.constant(10)
c1.get_shape()          #TensorShape([])
type(c1)                #tensorflow.python.framework.ops.Tensor
c1.eval()

# Rank 1 tensor (Vector)
c2 = tf.constant([10,20,30])
c2.get_shape()          #TensorShape([Dimension(3)])
c2.eval()               #array([10, 20, 30], dtype=int32)

c3 = tf.constant(np.array([20,30]))
c3.get_shape()          #TensorShape([Dimension(2)])
c3.eval()               #array([20, 30])

# Rank 2 tensor (Matrix)
c4 = tf.constant(np.array([[20,30],[50,60]]))
c4.get_shape()          #TensorShape([Dimension(2), Dimension(2)])
c4.eval()               #array([[20, 30],
                        #       [50, 60]])

#Give name to node not to Constants
c5 = tf.constant(100, name='constant')
c5.get_shape()          #TensorShape([])
c5.eval()               #100

#Using shape argument
c6 = tf.constant(10, shape=[3])
c6.eval()               #array([10, 10, 10], dtype=int32)

c7 = tf.constant(10.0, shape=[3])
c7.eval()               #array([ 10.,  10.,  10.], dtype=float32)

c77 = tf.constant(-1, shape=[2,3])
c77.eval()              #array([[-1, -1, -1],
                        #   [-1, -1, -1]], dtype=int32)

z1 = tf.zeros(5, tf.int32) ## default type is float; avoid by specifying type explicitly
z1.eval()               #array([0, 0, 0, 0, 0], dtype=int32)
type(z1)                #tensorflow.python.framework.ops.Tensor
z1.get_shape()          #TensorShape([Dimension(5)])

z2 = tf.zeros((2,2))
type(z2)
z2.get_shape()          #TensorShape([Dimension(2), Dimension(2)])
z2.eval()               #array([[ 0.,  0.],
                        #       [ 0.,  0.]], dtype=float32)

session.close()

































 