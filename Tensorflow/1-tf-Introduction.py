import tensorflow as tf

# without any explicit execution of operations
c1 = tf.constant(100)
type(c1)
c1.get_shape()
print(c1)     # getting only object details not values

#execute explicitly with sessions
session1 = tf.Session()
print(session1.run(c1))    # 100
session1.close()

#Execute impicitly with session
#More convinient to program 
session2 = tf.InteractiveSession()
print(c1.eval())   #eval is the shortcut for session.run() method
#100
session2.close()

