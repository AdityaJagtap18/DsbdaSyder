import numpy as np
import pandas as pd



df = pd.read_csv(Iris.csv)

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
axes[0,0].hist(df["SepalLengthCm"]);
axes[0,1].set_title("Distribution of Second Column")
axes[0,1].hist(df["SepalWidthCm"]);
axes[1,0].set_title("Distribution of Third Column")
axes[1,0].hist(df["PetalLengthCm"]);
axes[1,1].set_title("Distribution of Fourth Column")
axes[1,1].hist(df["PetalWidthCm"]);
