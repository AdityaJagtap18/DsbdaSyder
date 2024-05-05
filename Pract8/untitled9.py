import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("Titanic-Dataset.csv")
print(dataset)


#Distplot
sns.distplot(dataset['Fare'])

sns.jointplot(x='Age', y='Fare', data=dataset)

#Pairplot

sns.pairplot(data=dataset)

#RugPlot
sns.rugplot(dataset['Fare'])


sns.barplot(dataset,x='Age',y='Fare')