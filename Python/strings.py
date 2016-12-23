
#strings represent group of characters and is immutable by  nature
s1 = "abcdedf"
type(s1)

s1[0:3]
#'abc'

s1[::2]
#'acef'

s1[::3]
#'adf'

s2 = s1.replace("ab","xy")
# 'xycdedf'

s3 = s1.capitalize()
#'Abcdedf'

isinstance(s3, str)
#True

#convert string to list of characters
s4 = list(s1)
s4[0] = 'x'
s4
#s1 = "abcdedf"
#['x', 'b', 'c', 'd', 'e', 'd', 'f']

#list is converted to string(incorrect)
s5 = str(s4)
type(s5)
#"['x', 'b', 'c', 'd', 'e', 'd', 'f']"
# working, i dont know how... its invalid way to perform this conversion in python

#correct way of converting list to string
s6 = ''.join(map(str,s4))
type(s6)
#'xbcdedf'

help(map)

