import pandas as pd

df = pd.read_csv("weather_data.csv")

print("========================================================================")
print("Weather Data")
print(df)
print("========================================================================")

dfh= df.head()
print("Weather head")
print(dfh)
print("========================================================================")

dft= df.tail()
print("Weather tail")
print(dft)
print("========================================================================")

dfd = df.describe()
print("Weather Describe")
print(dfd)
print("========================================================================")


dfDrop = df.dropna(axis = 1)
df.dropna(how = 'all')
print("Df Drop")
print(dfDrop)
print("========================================================================")

dfTemp = df.dropna(subset=['temperature'])
print("Temprature Drop")
print(dfTemp)
print("========================================================================")

dfWind = df.dropna(subset=['windspeed'])
print("windspeed Drop")
print(dfWind)
print("========================================================================")

temp_mean = df['temperature'].mean()
df['temperature'] = df['temperature'].fillna(temp_mean)
print("Temp values with mean")
print(df)
print("========================================================================")


wind_mean = df['windspeed'].mean()
df['windspeed'] = df['windspeed'].fillna(temp_mean)
print("windspeed values with mean")
print(df)
print("========================================================================")


df['event'] = df['event'].fillna("No Event")
print("event values with mean")
print(df)
print("========================================================================")

print("Final Data Frame")
print(df)
print("========================================================================")


