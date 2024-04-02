# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the dataset and preprocess it, including encoding categorical variables.
2. Split the dataset into features (X) and the target variable (y), and then split them into training and testing sets.
3. Initialize and train a DecisionTreeRegressor model on the training data.
4. Evaluate the model's performance using Mean Squared Error (MSE) and R^2 Score, and make predictions on new data points.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MOHAMED HAMEEM SAJITH J
RegisterNumber:  212223240090
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Read the dataset
data = pd.read_csv("/content/Salary_EX7.csv")

# Display the first few rows of the dataset
data.head()

# Display information about the dataset
data.info()

# Check for missing values
data.isnull().sum()

# Encode the 'Position' column
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

# Display the first few rows of the modified dataset
data.head()

# Define features and target variable
x = data[["Position", "Level"]]
y = data["Salary"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Initialize the DecisionTreeRegressor
dt = DecisionTreeRegressor()

# Train the model
dt.fit(x_train, y_train)

# Make predictions
y_pred = dt.predict(x_test)

# Calculate Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)

# Print Mean Squared Error
print("Mean Squared Error:", mse)

# Calculate R^2 score
r2 = metrics.r2_score(y_test, y_pred)

# Print R^2 score
print("R^2 Score:", r2)

# Predict on a new data point
prediction = dt.predict([[5, 6]])
print("Predicted salary:", prediction)


```

## Output:
# 1. DATA:
![image](https://github.com/Sajith7862/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972360/2fbb5e13-28f9-4c34-91ac-4ac054f592df)

# 2. MEAN SQUARED ERROR:
![image](https://github.com/Sajith7862/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972360/dc9dffc0-6c4d-4bfa-b2f6-3b0af7fe34e2)

# 3. PREDICTION:
![image](https://github.com/Sajith7862/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972360/46cf970f-6d3d-418c-86b9-de9132a2d103)

# 4.CART ALGORITHM:
![image](https://github.com/Sajith7862/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972360/b6ae65d0-347f-4a03-8a38-c90189c60b49)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
