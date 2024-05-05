import pandas as pd
import numpy as np

df =pd.read_csv("Mall_Customers.csv")
print("Mall Customers csv")
print(df)
print("=======================================================================")

all_mean = df.mean(numeric_only = True)
print("All Mean")
print(all_mean)
print("=======================================================================")

all_std = df.std(numeric_only = True)
print("All Std")
print(all_std)
print("=======================================================================")

df_numeric = df.select_dtypes(include = [int,float])
all_median = df_numeric.median()
print("All Median")
print(all_median)
print("=======================================================================")

print("Min Values")
print(df.min())
print("=======================================================================")

print("Max Values")
print(df.max())
print("=======================================================================")

gk = df.groupby("Genre")
print(gk.first())
print("=======================================================================")



df =pd.read_csv("Iris.csv")
print("Iris csv")
print(df)
print("=======================================================================")

all_mean = df.mean(numeric_only = True)
print("All Mean")
print(all_mean)
print("=======================================================================")

all_std = df.std(numeric_only = True)
print("All Std")
print(all_std)
print("=======================================================================")

df_numeric = df.select_dtypes(include = [int,float])
all_median = df_numeric.median()
print("All Median")
print(all_median)
print("=======================================================================")

print("Min Values")
print(df.min())
print("=======================================================================")

print("Max Values")
print(df.max())
print("=======================================================================")

gk = df.groupby("Species")
print(gk.first())
print("=======================================================================")


iris_dataset = df["Species"] == "Iris-setosa"
print(df[iris_dataset].describe())
print("=======================================================================")


iris_dataset = df["Species"] == "Iris-versicolor"
print(df[iris_dataset].describe())
print("=======================================================================")


iris_dataset = df["Species"] == "Iris-virginica"
print(df[iris_dataset].describe())
print("=======================================================================")
