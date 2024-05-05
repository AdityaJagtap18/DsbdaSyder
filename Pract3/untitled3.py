import pandas as pd
import numpy as np
df = pd.read_csv("Mall_Customers.csv")
print("Mall Customer CSV")
print(df)
print("========================================================================")

dfSum = df.isnull().sum()
print("Null Sums")
print(dfSum)
print("========================================================================")

all_mean = df.mean(numeric_only = True)
print(all_mean)
print("========================================================================")

dfAge = df['Age'].mean()
print("Age mean")
print(dfAge)
print("========================================================================")

dfAIC = df['Annual Income (k$)'].mean()
print("Annual Income mean")
print(dfAIC)
print("========================================================================")

dfCID = df['CustomerID'].mean()
print("CustomerID mean")
print(dfCID)
print("========================================================================")


numeric_df = df.select_dtypes(include=[int,float])
all_median = numeric_df.median()
print("All median")
print(all_median)
print("========================================================================")

all_std = df.std(numeric_only = True)
print(all_std)
print("========================================================================")

print(df.min())
print("========================================================================")

print(df.max())
print("========================================================================")

gk=df.groupby("Genre")
print(gk.first())

print("========================================================================")


df = pd.read_csv("Iris.csv")
print("Iris CSV")
print(df)
print("========================================================================")

dfSum = df.isnull().sum()
print("Null Sums")
print(dfSum)
print("========================================================================")

all_mean = df.mean(numeric_only = True)
print(all_mean)
print("========================================================================")


numeric_df = df.select_dtypes(include=[int,float])
all_median = numeric_df.median()
print("All median")
print(all_median)
print("========================================================================")

all_std = df.std(numeric_only = True)
print(all_std)
print("========================================================================")

print(df.min())
print("========================================================================")

print(df.max())
print("========================================================================")

gk=df.groupby("Species")
print(gk.first())
print("========================================================================")

print("Iris-setosa")
iris_set1 = (df['Species']=="Iris-setosa")
print(df[iris_set1].describe())
print("========================================================================")

print("Iris-versicolor")
iris_set2 = (df['Species']=="Iris-versicolor")
print(df[iris_set2].describe())
print("========================================================================")


print("Iris-virginica")
iris_set3 = (df['Species']=="Iris-virginica")
print(df[iris_set3].describe())
print("========================================================================")

