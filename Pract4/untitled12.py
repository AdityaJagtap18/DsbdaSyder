import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


df=pd.read_csv("HousingData.csv")
df.head()

print(df.keys())

x = df.drop(columns=["MEDV"])  # Features
y = df["MEDV"] 

print(x.head())

print(x.shape, y.shape)

print(x.info())

print(x.describe())

print(y.info())

print(y.describe())

print(x.isnull().sum())

print(y.isnull().sum())

df = x 
df["target"] = y
print(df.head())

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True)
plt.show()

df = df[['RM', 'LSTAT', 'target']]

sns.pairplot(df)
plt.show()


x = df[['RM', 'LSTAT']]
y = df['target']

scaler = StandardScaler()

imputer = SimpleImputer(strategy="mean")  # Impute missing values with the mean
x_imputed = imputer.fit_transform(x)  # Impute feature data
y_imputed = y.fillna(y.mean())  # Impute

x_train, x_test, y_train, y_test = train_test_split(x_imputed, y_imputed, test_size=0.3, shuffle=True, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

model = LinearRegression(n_jobs=-1)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)

mean_absolute_error(y_test, y_pred)

mean_squared_error(y_test, y_pred)

sns.regplot(x=y_test, y=y_pred, scatter=True, line_kws={"color": "red"})
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Regression Plot: Actual vs Predicted")
plt.show()

