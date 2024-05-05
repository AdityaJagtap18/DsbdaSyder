import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("StudentPerformance.csv")
print(df)

new_df = df
col =['Maths_Score']
new_df.boxplot(col)

fig,ax= plt.subplots(figsize=(18,10))
ax.scatter(df['Placement_Score'],df['Placement offer count'])
plt.show()

q1=np.percentile(df['Maths_Score'], 25)
q3=np.percentile(df['Maths_Score'], 75)
IOR=q3-q1
print("IQR Value")
print(IOR)

lwr_bound = q1-(1.5*IOR)
uppr_bound = q3+(1.5*IOR)
print("Upper Bound: ",uppr_bound," Lower Bound: ",lwr_bound)
iden_outlier = np.where((df[col]<lwr_bound)|df[col]>uppr_bound)
print(iden_outlier)


