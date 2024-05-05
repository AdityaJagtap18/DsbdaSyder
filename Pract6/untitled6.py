import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('IrisDataset.csv')

print(dataset.head())

gk = dataset.groupby("species")
print(gk.first())

from sklearn.model_selection import train_test_split

x= dataset.iloc[:,:4].values

y= dataset['species'].values

x_train ,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(cm)


df=pd.DataFrame({'Real Values':y_test,'Prediected values':y_pred})

print(df)