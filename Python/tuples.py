#tuple is an immutable list 
tuple1 = (10, 30, 20, 40)
type(tuple1)
#tuple

tuple1
#(10, 30, 20, 40)

tuple1[0]
# 10

tuple1[0:2] 
#(10, 30)

tuple1[0:] 
#(10, 30, 20, 40)

tuple1[:3] 
#(10, 30, 20)

tuple1[::2] 
#(10,20)

len(tuple1)

# Accessing tuple items
for x in tuple1:
    print x
    
tuple2 = (10,20)
x,y = tuple2
x
y
#10
#20

