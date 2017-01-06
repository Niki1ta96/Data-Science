import tensorflow as tf

session = tf.InteractiveSession()

a = tf.Variable(10)
type(a)                 #tensorflow.python.ops.variables.Variable
a.get_shape()           #TensorShape([])

aa = tf.constant(10)
type(aa)                #tensorflow.python.framework.ops.Tensor
aa.get_shape()          #TensorShape([])

b = tf.Variable(tf.zeros(5))
type(b)
b.get_shape()           #TensorShape([Dimension(5)])

c = tf.Variable(tf.zeros((2,3)))
c.get_shape()           #TensorShape([Dimension(2), Dimension(3)])
c.eval()                #Error, for variables we need to initialize first

#  variables need to be initialised, but constants don't need to
session.run(tf.initialize_all_variables())
a.eval()                #10
b.eval()                #array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)
c.eval()                #array([[ 0.,  0.,  0.],
                       #        [ 0.,  0.,  0.]], dtype=float32)

# Evaluate all variables at once
session.run([a,b,c])    #[10, 
                        #array([ 0.,  0.,  0.,  0.,  0.], dtype=float32), 
                        #array([[ 0.,  0.,  0.],
                        #      [ 0.,  0.,  0.]], dtype=float32)]
                        
seed = tf.set_random_seed(999)

#Floating points variables
d1 = tf.Variable(tf.random_uniform((10,)))

## I didn't use numpy's random, but tf's random to make use of tf functionality
d2 = tf.Variable(tf.random_uniform((10,),0,2))

#Integer random variables
d3 = tf.Variable(tf.random_uniform(shape=(10,), maxval=100, minval=1, dtype = tf.int32))

#
session.run(tf.initialize_all_variables())
d1.eval()              # 10 random uniform numbers generated
d2.eval()               # 10 random float numbers generated
d3.eval()               #10 numbers bet 1 to 100 integer type generated

d = tf.constant(10)
e = tf.Variable(d+10)
f = tf.add(e, tf.constant(1))

session.run(tf.initialize_all_variables())
session.run([d,e,f])        #[10, 20, 21]

update = e.assign(e+10)
update.eval()               #30

session.run([d,e,f])        #[10,30,31]

session.run([d,e,f,update]) #[10, 40, 41, 40]




























                                
