import pandas as pd
from HelperFunctions import Helper


## TODO: [1] Read the Data.csv ##
## Start Code ## (≈1 line)

global df
df = pd.read_csv('Houses.csv')


## End Code ##

## TODO: [2] Clean the DataFrame Features ##
## TODO: [2.1] Remove Columns with String Values from your DataFrame ##
## TODO: [2.2] Change the `date_added` feature to datetime format ##
## TODO: Use RemoveStringColumns() & GetDateFeature() ##
## Start Code ## (≈3 lines)

mylist = ["location", "city", "purpose"]
Helper.Remove_My_String_Columns(df, mylist)
Helper.GetDateFeature(df, "date_added")

## End Code ##

## TODO: [3] Build the Simple Linear Regression Models ##
## TODO: Use LinearRegressionModel() ##
## Start Code ##

F = input("Please Enter Your Feature Name : ")
G = input("Please Enter Your Ground truth Name : ")
Helper.Linear_Regression_Model(F, G)

## End Code ##
