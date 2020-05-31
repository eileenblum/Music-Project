from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from sklearn.neighbors import KNeighborsClassifier
# Make the model
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the model
knn.fit(X_train,y_train.ravel())

# Make a prediction on our train set
y_predict = knn.predict(X_train)
# calculate the accuracy here
print("Our model has a ",
      np.round(sum(y_predict == y_train.ravel())/len(y_train)*100,2),
      "% accuracy on the training set")

# Cross-Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

max_neighbors = 100
# make the kfold object
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state = 440)

# make an empty array that will hold all of our accuracy measures
accs = np.empty([max_neighbors,n_splits])

# We'll go from 1 to 10 neighbors
for i in range(1,max_neighbors+1):
    # make knn
    knn = KNeighborsClassifier(n_neighbors = i)
    # We'll get the accuracy for each split
    j = 1
    for train_index, test_index in kfold.split(X_train,y_train):
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train[test_index]
        
        clone_knn = clone(knn)
        clone_knn.fit(X_train_folds,y_train_folds.ravel())
        y_pred = clone_knn.predict(X_test_fold)
        accs[i-1,j-1] = np.round(sum(y_pred == y_test_fold.ravel())/len(y_pred),4)
        j = j+1

# Plot how the accuracy changes
sns.set_style("whitegrid")

plt.figure(figsize = (12,8))
plt.plot(range(1,max_neighbors+1),np.mean(accs*100,axis = 1),'-o')

plt.xticks(range(1,max_neighbors+1),fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel("Number of Neighbors",fontsize = 16)
plt.ylabel("Accuracy (in %)",fontsize = 16)
plt.show()

#interpret the model
##What does the best word model mean? 
##Word2vec can most accurately determine semantic similarity of a word based on x number of its neighbors
##Song lytics do tend to utilize semantically similar words based on genre?
