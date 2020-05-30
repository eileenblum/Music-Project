import pandas as pd
import numpy as np
import sklearn
## For plotting
import matplotlib.pyplot as plt
import seaborn as sns
## This sets the plot style
## to have a grid on a white background
sns.set_style("whitegrid")
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
#data_corpus=["guru99 is the best sitefor online tutorials. I love to visit guru99."]


file = open('lorem_Ipsum.txt','rt')
latin = file.read()

#Convers a text file into a list of strings containing its word
latin_list = latin.split()

vocabulary=vectorizer.fit(latin_list)
X= vectorizer.transform(latin_list)
Y = X.toarray()
#print(Y.shape)
#Y.max()
V = vocabulary.get_feature_names()
#print(V)


Freq = np.zeros(len(V))
temp = 0
for v in range(len(V) ):
    for j in range( len(latin_list) ):
        if(Y[j,v] != 0):
           # print(j,v)
            temp += 1
    Freq[v] = temp
    temp = 0    
"""        
plt.figure(figsize=(10,10))
plt.hist(Freq)
plt.show()
"""


fig, axes = plt.subplots(5, 1, figsize = (12,30))
ax = axes.ravel()
for i in range(5):

    ax[i].hist(Freq, bins = 20 + 4*i, color = 'red', alpha = .5)
    ax[i].set_yticks(())

ax[0].legend(['malignant', 'benign'], loc = 'best')
    
fig.tight_layout()

















