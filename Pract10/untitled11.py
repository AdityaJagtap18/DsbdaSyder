import numpy as np
import pandas as pd


csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(csv_url, header = None)
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
df = pd.read_csv(csv_url, names = col_names)

dfh = df.head()

print(dfh)

column = len(list(df))

dfi =df.info()

print(dfi)

print(np.unique(df['Species']))

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes[0,0].set_title("Distribution of First Column")
axes[0,0].hist(df["Sepal_Length"]);
axes[0,1].set_title("Distribution of Second Column")
axes[0,1].hist(df["Sepal_Width"]);
axes[1,0].set_title("Distribution of Third Column")
axes[1,0].hist(df["Petal_Length"]);
axes[1,1].set_title("Distribution of Fourth Column")
axes[1,1].hist(df["Petal_Width"]);