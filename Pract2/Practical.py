import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("StudentPerformance.csv")

print("Student Performance CSV")
print(df)

new_df = df
col = ['Maths_Score']
new_df.boxplot(col)

fig,ax = plt.subplots(figsize = (18,10))
ax.scatter(df['Placement_Score'],df['Placement offer count'])
plt.show()

q1 = np.percentile(df['Maths_Score'], 25)
q3 = np.percentile(df['Maths_Score'], 75)
IQR = q3-q1
print("IQR: ",IQR)

lwr_bound = q1-(1.5*IQR)
uppr_bound = q3+(1.5*IQR)

print("Lower Bound: ",lwr_bound,"Upper Bound: ",uppr_bound)

outliers = np.where((df[col]<lwr_bound)|(df[col]>uppr_bound))
print("Outliers are: ",outliers)

