import pandas as pd
import numpy as np
import scipy
import gensim
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import word2vec
from gensim.models import Word2Vec
sns.set_style("whitegrid")

#Roberto may not need to read in the file
with open("word2vec.RM_model", 'rb') as f:
    rm_model = f.read()

#What we really need is to convert model data into a pandas dataframe
#start by making a numpy array
rmarray = np.array(rm_model)
np.shape(rmarray)


#then convert to pandas
word2vec_df = pd.DataFrame(rmarray)

import time #only need this to record runtime
#decision tree classifier
word2vec_clf = DecisionTreeClassifier(random_state = 440)
start_time = time.time()
# Fit the model
word2vec_clf.fit(list(Pro_Lyrics_train.s_vector, type_train.type))
print("Time taken to fit the model with word2vec vectors: " + str(time.time() - start_time))

# Plot the fitted tree
plt.figure(figsize = (10,10))
fig = word2vec_clf.fit(df[['x1','x2']], df.y)
word2vec.plot_word2vec(fig,filled = True)
plt.show()

#generate classification report
from sklearn.metrics import classification_report
test_features_word2vec = []
for index, row in X_test.iterrows():
    model_vector = np.mean([RM_model[token] for token in row['stemmed_tokens']], axis=0)
    if type(model_vector) is list:
        test_features_word2vec.append(model_vector)
    else:
        test_features_word2vec.append(np.array([0 for i in range(1000)]))
test_predictions_word2vec = clf_decision_word2vec.predict(test_features_word2vec)
print(classification_report(Y_test['genre'],test_predictions_word2vec))