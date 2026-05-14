import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "Student_Hours": [1, 2, 3, 4, 5],
    "Marks": [20, 40, 60, 80, 100]
}

df = pd.DataFrame(data)
print("Complete dataset.")
print(df)

print("\n-------------------------------Dataset Infotrmation-------------------------------------------------")
print(df.info())

print("\n-------------------------------Statistical summary of the dataset-----------------------------------")
print(df.describe())

# SELECT FEATURES AND LABELS
print("\n-------------------------------Selecting Features and Labels----------------------------------------")
x = df[["Student_Hours"]]
y = df[["Marks"]]
print("Features:", x)
print("Labels:", y)

print("\n-------------------------------Splitting the dataset into training and testing sets------------------")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print("Training set features:", x_train)
#print("Training set labels:", y_train)
#print("Testing set features:", x_test)
#print("Testing set labels:", y_test)

#-------------------------------Training the Linear Regression model-----------------------------------
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print("\n-------------------------------predicted values-----------------------------------------")
print(predictions)

print("\n-------------------------------Actual values-----------------------------------------")
print(y_test.values)

print("\n-------------------------------Model Coefficients and Intercept-----------------------------------------")
print("\nModel Coefficient (Weight):")
print(model.coef_)

print("\nModel Intercept (Bias):")
print(model.intercept_)