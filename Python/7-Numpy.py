import numpy as np

# Regular python list
height = [10,20,30]
weight = [100,200,300]

weight + height
#[100, 200, 300, 10, 20, 30]

height ** 2
#Error

height >  20
#True
# showing only for last element of list 

#Numpy array
np_height = np.array(height)
np_weight = np.array(weight)

np_height + np_weight
# array([110, 220, 330])

np_height ** 2
#array([100, 400, 900])

np_height > 20
#array([False, False,  True], dtype=bool)

np_height[np_height > 20]
#array([30])

list1 = [10, True, 'abc']
array1 = np.array(list1)
#array(['10', 'True', 'abc'], dtype='|S11')

## S11 is a data type descriptor, it means internally array holds the string of length 11
## | pipe symbol for byteorder flag; in this case there is no byte order flag needed
# So, its set to pipe |, meaning not applicable


np.mean(height)
# 20.0

a1 = np.array([1,2,3],[4,5,6])
# error data types not understood

a1 = np.array([[1,2,3],[4,5,6]])
# its working

a1[1,1]
#5

a1[1:2,1]
#array([5])



