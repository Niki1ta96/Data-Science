import os
import pandas as pd

os.getcwd()
os.chdir("C:\\Users\\nikita.jaikar\\Documents\\GitHub\\Data-Science\\Python")

#read the data from csv file into data frame
titanic_train = pd.read_csv("C:\\Users\\nikita.jaikar\\Documents\\GitHub\\Data-Science\\Data\\titanic_test.csv")
titanic_train.shape

#provide the schema related information and memory requirements
titanic_train.info
titanic_train.dtypes

#show sample data
titanic_train.head()
titanic_train.tail()

#provide summary statistics
titanic_train.describe()

#access the frame content by column/columns
titanic_train["Age"]
titanic_train.Age
titanic_train[["PassengerId", "Fare"]]

#dropping a column
titanic_train1 = titanic_train.drop('Fare', axis = 1) #axis =1 means column
titanic_train1.info

#slicing rows of frame
titanic_train[0:4]
titanic_train[:7]
titanic_train[-3:]
#titanic_train[[0,3]]  #working 

#slicing rows based on condition
titanic_train[titanic_train.Age > 50]

#slicing subset of rows and columns 
titanic_train.iloc[0,3]
titanic_train.iloc[0:3,0:3]
titanic_train.iloc[0:3,:]
titanic_train.iloc[:,0]
titanic_train.iloc[[0,2],:]
titanic_train.iloc[:,[0,2,6]]
titanic_train.loc[0:4,['Age']]

#setting a column as index column
titanic_train.set_index('PassengerId')
titanic_train.set_index('PassengerId', inplace = True)

#resetting index column
titanic_train.reset_index()
titanic_train.reset_index(inplace = True)
