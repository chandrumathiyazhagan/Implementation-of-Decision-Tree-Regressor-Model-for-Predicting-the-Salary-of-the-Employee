# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the packages that helps to implement Decision Tree.

2. Download and upload required csv file or dataset for predecting Employee Churn

3. Initialize variables with required features.

4. And implement Decision tree classifier to predict Employee Churn

## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: M.CHANDRU

RegisterNumber:  212222230026

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
data=pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(18,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```
## Output:

### DATASET:
![Screenshot 2024-04-03 122515](https://github.com/chandrumathiyazhagan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393023/bf3b8489-a38e-4d79-8c5d-9e4d3b627135)

### MSE:
![Screenshot 2024-04-03 122120](https://github.com/chandrumathiyazhagan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393023/e3f634a9-580c-44dd-b080-c6bf77783190)

### R2(variance):
![Screenshot 2024-04-03 122112](https://github.com/chandrumathiyazhagan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393023/7592574e-a383-48de-b4de-86b61ee931d2)

### DATA PREDICTION:
![Screenshot 2024-04-03 122141](https://github.com/chandrumathiyazhagan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393023/827a2630-614d-433c-b87d-18d7d2be6050)

### DECISION TREE:
![Screenshot 2024-04-03 122101](https://github.com/chandrumathiyazhagan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393023/7568da3d-8d61-4c23-974b-2f2732ed8705)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
