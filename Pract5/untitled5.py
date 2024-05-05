import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv("Social_Network_Ads.csv")
print("Social Network Ads csv")
print(dataset)
print("========================================================================")

dfNull = dataset.isnull().sum()
print("Null Values")
print(dfNull)
print("========================================================================")

x= dataset.iloc[:,[2,3]].values
y= dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state= 0)

print(x_train[:3])
print("========================================================================")

print(x_test[:3])
print("========================================================================")

print(y_train[:3])
print("========================================================================")

print(y_test[:3])
print("========================================================================")

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

print(x_train[:3])
print("========================================================================")

print(x_test[:3])
print("========================================================================")

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0 ,solver='lbfgs')

#'lbfgs' (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print(x_test[:10])
print("========================================================================")

print(y_pred[:20])
print(y_test[:20])
print("========================================================================")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Make sure these are properly initialized
X_set, y_set = x_train, y_train

# Generate a mesh grid for contour plot
X1, X2 = np.meshgrid(
    np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, step=0.01),
    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, step=0.01),
)

# Ensure proper reshaping of classifier predictions for contour plot
predictions = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
predictions = predictions.reshape(X1.shape)

# Create the contour plot with appropriate color mapping
plt.contourf(X1, X2, predictions, alpha=0.75, cmap=ListedColormap(['red', 'green']))

# Set x and y limits for the plot
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Scatter plot for the training set with correct color mapping
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        color=ListedColormap(['red', 'green'])(i),  # Correct color mapping
        label=j,
    )

# Set plot title and axis labels
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Feature 1')  # Change this to your feature name
plt.ylabel('Feature 2')  # Change this to your feature name
plt.legend()
plt.show()

