import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv")
df.head()
df.info()
df.describe()
df.count()

dfn =df.isnull().sum()

print(dfn)

df1=df.dropna()
df1

df1.isnull().sum()

df2=df1.drop(['Name', 'SibSp','Ticket','Pclass','PassengerId','Parch','Cabin','Embarked'], axis=1)

print(df2.count())

print(df2)

sns.set_theme(style="ticks", color_codes=True)


# count plot on single categorical variable
sns.countplot(x ='Sex', data = df2)
 
# Show the plot
plt.show()


sns.countplot(x ='Survived', data = df2)

# Show the plot
plt.show()

sns.set_style("whitegrid")

sns.boxplot(x = 'Sex', y = 'Age', data = df2)


sns.boxplot(x = df2['Sex'],
            y = df2['Age'],
            hue = df2['Survived'],
            palette = 'Set2')