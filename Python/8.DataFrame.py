import os
import pandas as pd
import random as r # can also use numpy for random() function


os.chdir("C:\\Users\\nikita.jaikar\\Documents\\GitHub\\Data-Science\\Data")
os.getcwd()

titanic_train = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")

titanic_train.shape
# Dimensions (891, 12)

# convert to category data type
titanic_train.Pclass = titanic_train.Pclass.astype('category') # casting to categorical type
# Before :  dtype: int64
#After   : Name: Pclass, dtype: category
#          Categories (3, int64): [1, 2, 3]

type(titanic_train)
#pandas.core.frame.DataFrame

titanic_train.info
# Summary info on dataset

titanic_train.dtypes  # Type of every column
# PassengerId       int64
# Survived          int64
# Pclass         category
# Name             object
# Sex              object
# Age             float64
# SibSp             int64
# Parch             int64
# Ticket           object
# Fare            float64
# Cabin            object
# Embarked         object
# dtype: object


# Data Exploration 
titanic_train.head()
titanic_train.tail()
titanic_train.head(3)

# Equivalent to summary(train) in R
titanic_train.describe()
# Gives deatils like mean, median min, max and sd for each columns
# Thus gives Central tendency and Spread 
# Show stat for 'label' as it is also converted to numerical

#Accessing multiple columns by name or index
titanic_train[[0, 1]]
titanic_train[['Age','Pclass']]

#Accessing one columns by name or index
titanic_train[[0]]
titanic_train[['Pclass']]

# Accessing row based on conditiion 
titanic_train[titanic_train.Age>70]

#Setting index 
titanic_train.set_index('PassengerId')
titanic_train.set_index('PassengerId', inplace = True)
titanic_train.reset_index()
titanic_train.reset_index(inplace = True)

titanic_train.loc[2:]

#Randomdly Predict the survival column
titanic_test.Survivor = r.randrange(0,1) # 0
# Vectorization NOT by default on basic data types
# Vectorization only in np.array from numpy package









