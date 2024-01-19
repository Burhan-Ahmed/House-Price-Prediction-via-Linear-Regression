import pandas as pd
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt


dataset=pd.read_csv("train.csv")
test_set=pd.read_csv("test.csv")

X=dataset[["BedroomAbvGr","FullBath","1stFlrSF"]]
Y=dataset["SalePrice"]

reg= linear_model.LinearRegression()
reg.fit(X,Y)

print("Done Building Model")

test_cases=test_set[["BedroomAbvGr","FullBath","1stFlrSF"]]
testing = reg.predict(test_cases)

print(testing)

print('Coefficients:', reg.coef_)
print('Intercept:', reg.intercept_)