import numpy as np

a1 = np.array([[1,2,3],[4,5,6]])
a1.shape
#(2L, 3L)
a1[1,1]
#5

#use colon to get all values across dimension
a1[1,:]
#array([4, 5, 6])

a1[:,1]
#array([2, 5])

#colon can also be used for range
a1[0:2,1]
#array([2, 5])

#create zero matrix of size 2 by 3
a2 = np.zeros((2,3), int)
#array([[0, 0, 0],
#      [0, 0, 0]])

a2.shape
#(2L, 3L)

#create one matrix of size 3 by 2
a3 = np.ones((2,3), int) # by default float type
a3.shape
#(2L, 3L)

#create an identity matrix of size 3 by 3
a4 = np.eye(3,3,dtype = int) # by default float type
#array([[1, 0, 0],
#       [0, 1, 0],
#       [0, 0, 1]])

a4.shape
#(3L, 3L)

#reshape the matrix
a5 = np.array([       #np.array() convert list data to strings
               [1,2],
               [3,4],
               [5,6]
               ])

a5.shape
#(3L, 2L)

a5.reshape((2,3))
#array([[1, 2, 3],
#       [4, 5, 6]])

tmp1 = a5.reshape(1, 6)
#array([[1, 2, 3, 4, 5, 6]])

type(tmp1)
#numpy.ndarray

#Reshape a matrix into a single row, figuring out the correct number of columns
a6 = a5.reshape((1,-1))
#array([[1, 2, 3, 4, 5, 6]])
a6.shape
#(1L, 6L)
type(a6)
#numpy.ndarray

tmp2 = a6.reshape(2,3)
#array([[1, 2, 3],
#       [4, 5, 6]])

#getting useful statistics on matrix
a1
#array([[1, 2, 3],
#       [4, 5, 6]])
a1.max(axis =0)    # axis = 0 ---> ROWS
#array([4, 5, 6])
a1.max(axis = 1)  # axis = 1 ---> Columns
#array([3, 6])

a1.mean(axis = 0)
#array([ 2.5,  3.5,  4.5])
a1.mean(axis = 1)
#array([ 2.,  5.])
a1.std(axis = 0)
#array([ 1.5,  1.5,  1.5])
a1.std(axis = 1)
#array([ 0.81649658,  0.81649658])

#Element wise operations on matrices
a7 = np.array([[1,2],[3,4]])
#array([[1, 2],
#       [3, 4]])

a8 = np.array([[1,1],[2,2]])
#array([[1, 1],
#       [2, 2]])

a7 + a8
#array([[2, 3],
#       [5, 6]])

a7 * a8    # respective element value multiplication -- not useful in real world
#array([[1, 2],
#       [6, 8]])

#matrix multiplication, solving linear euations
a7.dot(a8)   # dot product for matrix multiplication 
#array([[ 5,  5],
#       [11, 11]])

#matrix transpose
a7.T
#array([[1, 3],
#       [2, 4]])
