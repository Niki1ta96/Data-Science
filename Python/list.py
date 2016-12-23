
#list is an indexed container that holds heterogeneous elements
list1 = [10, 30, 20, 40]
type(list1)
list1

#create a list with elements in the range of 1 to 10 with step size of 1
list2 = range(1,10,1)
list2
type(list2)

#sliced access to elements of list

list1[0]
#10 

list1[0:2]
#10, 30

list1[0:]
#[10, 30, 20, 40]

list1[:3]
#[10, 30, 20]

list1[0::2]
#[10, 20]

list1[0] = 100
list1
#[100, 30, 20, 40]

#modifying the contents of list
list3 = []
list3
list3.append(10)
#[10]

list3.append(20)
#[10,20]

list3.insert(3, 70)
#[10,20,70]

list3.append(True)
#[10, 20, 70, True]

list3.append(list1)
#[10, 20, 70, True,[10, 30, 20, 40]]

#sort the elements of list1
list1.sort()
#[20, 30, 40, 100]

len(list1)

#iterate through the list elements
for x in list1:
    print x
#20
#30
#40
#100
