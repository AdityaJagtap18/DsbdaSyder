
import pandas as pd

df = pd.read_csv("weather_data.csv")

dfh =df.head()
print(dfh)
print("==================")

dft = df.tail()
print(dft)
print("==================")


dfd = df.describe()
print(dfd)
print("==================")


missing_data = df.isnull()
print(missing_data)
print("==================")

df.dropna(axis =1)

df.dropna(how='all')
print("no nulls")
print(df)
print("==================")

dftemp = df.dropna(subset=['temperature'])
print("temp null")
print(dftemp)

print("==================")
dfwind = df.dropna(subset=['windspeed'])
print("windspeed null")
print(dfwind)

print("==================")
mean_temp = df['temperature'].mean()
df['temperature'] = df['temperature'].fillna(mean_temp)
print(df)

print("==================")
mean_temp = df['windspeed'].mean()
df['windspeed'] = df['windspeed'].fillna(mean_temp)
print(df)

print("==================")
df.dropna(subset=['event'])
df['event'] = df['event'].fillna("No Event")
print(df)

