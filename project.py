# Erdos Bootcamp project: Matelinguistica
# Eileen Blum (U Rutgers) and Roberto Hernandez Palomares (OSU)

#References: 
# guru99 website is a major reference for this code
# also  https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# and https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
import pandas as pd
import numpy as np
import os
import string
import nltk
import gensim
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import word2vec

from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import abc
from nltk.corpus import stopwords
from textblob import Word
from gensim.models import Word2Vec
sns.set_style("whitegrid")


##########################  1  #################################################
############# Loading the data #################################################
################################################################################
#establish working directory:
#os.chdir(r'C:\Users\hprob\Desktop\ErdosProjectMay2020\Sample_project\Music-Project-master\Music-Project-master\Lyrics\metal_lyrics')
#os.getcwd()
#entries = os.scandir(r'metal_lyrics')
#entries

###############  Gather the Metal Files  ################################
#Make sure to change this path to where YOU are storing lyrics in your computer    
Metal_name_and_text = {}
# Walking a directory tree and printing the names of the directories and files
for dirpath, dirnames, files in os.walk(r'C:\Users\hprob\Desktop\ErdosProjectMay2020\Sample_project\Music-Project-master\Music-Project-master\Lyrics\metal_lyrics'):
    print(f'Found directory: {dirpath}')
    for file_name in files:
        os.listdir()
        print("song:", file_name)
        with open(dirpath + r'\\' + file_name, "r", errors='ignore') as target_file:
            Metal_name_and_text[file_name] = target_file.read()
    Metal_data = (pd.DataFrame.from_dict(Metal_name_and_text, orient='index')
            .reset_index().rename(index=str, columns={'index': 'song_name', 0: 'lyrics'}))

Type = np.zeros(shape=(len(Metal_name_and_text),1 ))
Metal_data['type'] = Type


###############  Gather the Reggae Files  ###############################
Reggae_name_and_text = {}
DFReggae_name_and_text = pd.DataFrame([['delete','delete','delete','delete','delete']],
                                      columns= ['ARTIST_NAME', 'ARTIST_URL', 'SONG_NAME', 'SONG_URL', 'LYRICS'])

# Walking a directory tree and printing the names of the directories and files
for dirpath, dirnames, files in os.walk(r'C:\Users\hprob\Desktop\ErdosProjectMay2020\Sample_project\Music-Project-master\Music-Project-master\Lyrics\azlyrics-scraper'):
    for file_name in files:
        Reggae_name_and_text[file_name] = pd.read_csv(r'C:\Users\hprob\Desktop\ErdosProjectMay2020\Sample_project\Music-Project-master\Music-Project-master\Lyrics\azlyrics-scraper' + r'\\' + file_name, error_bad_lines=False) 
        temp = pd.read_csv(r'C:\Users\hprob\Desktop\ErdosProjectMay2020\Sample_project\Music-Project-master\Music-Project-master\Lyrics\azlyrics-scraper' + r'\\' + file_name, error_bad_lines=False)
        DFReggae_name_and_text = DFReggae_name_and_text.append(temp)  

del DFReggae_name_and_text['ARTIST_NAME']
del DFReggae_name_and_text['ARTIST_URL']
del DFReggae_name_and_text['SONG_URL']
DFReggae_name_and_text = DFReggae_name_and_text.reset_index()
DFReggae_name_and_text = DFReggae_name_and_text.drop( index = 0)
del DFReggae_name_and_text['index']

len(DFReggae_name_and_text)
Type = np.ones(shape=(len(DFReggae_name_and_text), 1 ))

Reggae_data = DFReggae_name_and_text.copy()

Reggae_data['type'] = Type####################################
Reggae_data.columns = ['song_name', 'lyrics', 'type']
del Type


##########################  2  #################################################
############# Preprocessing the Data ###########################################
################################################################################

##############  Unionizing data ####
#input pds Metal_data and Reggae_data
subMetal_data = Metal_data.sample(len(DFReggae_name_and_text))

Lyrics = subMetal_data.copy()
Lyrics = Lyrics.append(Reggae_data)  #<------------this is the data we will work with
Lyrics = Lyrics.dropna()
##########################################

############  Spliting into train and test ############
Lyrics_train, Lyrics_test, type_train, type_test = train_test_split(Lyrics[['song_name','lyrics']],
                                                                    Lyrics[['type']], 
                                                                    stratify=Lyrics[['type']], test_size=0.1)
#######################################################

################# Cleaning ##############################
stop = stopwords.words('english') #importing ENglish stop words
#cleaning and lemmatizing Lyrics_train:
Pro_Lyrics_train = pd.DataFrame(Lyrics_train).copy()
Pro_Lyrics_train['lyrics'] = Pro_Lyrics_train['lyrics'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))#All the rows of the text in the data frame is checked for string punctuations, and these are filtered
Pro_Lyrics_train['lyrics'] = Pro_Lyrics_train['lyrics'].str.replace('[^\w\s]','')#REmoves dots using regular expressions
Pro_Lyrics_train['lyrics'] = Pro_Lyrics_train['lyrics'].apply(lambda x:' '.join(x.lower() for x in x.split())) #Converting text to lower case
Pro_Lyrics_train['lyrics'] = Pro_Lyrics_train['lyrics'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))#Digits are removed from the text
Pro_Lyrics_train['lyrics'] = Pro_Lyrics_train['lyrics'].apply(lambda x: ' '.join(x for x in x.split() if not x in stop))#Stop words are removed at this stage
Pro_Lyrics_train['lyrics'] = Pro_Lyrics_train['lyrics'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))#Words are filtered now, and different form of the same word is removed using lemmatization
Pro_Lyrics_list = []
for i in Pro_Lyrics_train['lyrics']:
     li = list(i.split(" "))
     Pro_Lyrics_list.append(li)	
     
Pro_Lyrics_train['words'] = Pro_Lyrics_list #Adding the list of words/lemmas used for every song     



#cleaning and lemmatizing Lyrics_test:
Pro_Lyrics_test = pd.DataFrame(Lyrics_test).copy()
Pro_Lyrics_test['lyrics'] = Pro_Lyrics_test['lyrics'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))#All the rows of the text in the data frame is checked for string punctuations, and these are filtered
Pro_Lyrics_test['lyrics'] = Pro_Lyrics_test['lyrics'].str.replace('[^\w\s]','')#REmoves dots using regular expressions
Pro_Lyrics_test['lyrics'] = Pro_Lyrics_test['lyrics'].apply(lambda x:' '.join(x.lower() for x in x.split())) #Converting text to lower case
Pro_Lyrics_test['lyrics'] = Pro_Lyrics_test['lyrics'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))#Digits are removed from the text
Pro_Lyrics_test['lyrics'] = Pro_Lyrics_test['lyrics'].apply(lambda x: ' '.join(x for x in x.split() if not x in stop))#Stop words are removed at this stage
Pro_Lyrics_test['lyrics'] = Pro_Lyrics_test['lyrics'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))#Words are filtered now, and different form of the same word is removed using lemmatization
Pro_Lyrics_test_list = []
for i in Pro_Lyrics_test['lyrics']:
     li = list(i.split(" "))
     Pro_Lyrics_test_list.append(li)	
Pro_Lyrics_test['words'] = Pro_Lyrics_test_list #Adding the list of words/lemmas used for every song
################################################################################
#######  END OF PREPROCESSING  #################################################
################################################################################



######################  3  #####################################################
############# Training Phase ###################################################
################################################################################

##########################  Training a couple models  ###################
#Training a model with the vocabulary from Pro_Lyrics_list
RM_model = Word2Vec(Pro_Lyrics_list, min_count=2, size=150, workers=15, window=15)
print()
#Saving the model
RM_model.save("word2vec.RM_model")
RM_model.save("RM_model.bin")
print()

#Training a model with the imported vocabulary from abc.sents()
abc_model = gensim.models.Word2Vec(abc.sents(), min_count=2, size=150, workers=15, window=15)
print()
#Saving the model
abc_model.save("word2vec.abc_model")
abc_model.save("abc_model.bin")
print()


####################### Storing vectors gen by models #################
# Store the vectors for train data in following file
### Finish <----------------------------------------------------------------------------------------Incomplete
#word2vec_filename = OUTPUT_FOLDER + 'train_review_word2vec.csv'
#RM_vectors_filename = r'C:\Users\hprob\Desktop\ErdosProjectMay2020\Sample_project\RM_vectors.csv'
#with open(RM_vectors_filename, 'w+') as word2vec_file:
#    for index, row in Lyrics_train.iterrows():
#        model_vector = (np.mean([RM_model[token] for token in row['lyrics']], axis=0)).tolist()
#        if index == 0:
#            header = ",".join(str(ele) for ele in range(1000))
#            word2vec_file.write(header)
#            word2vec_file.write("\n")
#        # Check if the line exists else it is vector of zeros
#        if type(model_vector) is list:  
#            line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
#        else:
#            line1 = ",".join([str(0) for i in range(1000)])
#        word2vec_file.write(line1)
#        word2vec_file.write('\n')
### Finish <----------------------------------------------------------------------------------------Incomplete
################################################################################
#######  END OF Training phase  #################################################
################################################################################






############################### 4 ##############################################
################## Data Analysis phase #########################################
################################################################################

##### Similarity testing and playing around ################
#word: flesh
abc_voc = list(abc_model.wv.vocab)
print("Most similar to flesh according to abc_model: ")
abc_data = abc_model.wv.most_similar('flesh')
print(abc_data)

RM_voc = list(RM_model.wv.vocab)
print("Most similar to flesh according to RM_model: ")
RM_data = RM_model.wv.most_similar('flesh')
print(RM_data)

### word: worm
abc_voc = list(abc_model.wv.vocab)
print("Most similar to worm according to abc_model: ")
abc_data = abc_model.wv.most_similar('worm')
print(abc_data)

RM_voc = list(RM_model.wv.vocab)
print("Most similar to worm according to RM_model: ")
RM_data = RM_model.wv.most_similar('worm')
print(RM_data)

### word: plastic
abc_voc = list(abc_model.wv.vocab)
print("Most similar to plastic according to abc_model: ")
abc_data = abc_model.wv.most_similar('plastic')
print(abc_data)

RM_voc = list(RM_model.wv.vocab)
print("Most similar to plastic according to RM_model: ")
RM_data = RM_model.wv.most_similar('plastic')
print(RM_data)

### word: hell
abc_voc = list(abc_model.wv.vocab)
print("Most similar to hella ccording to abc_model: ")
abc_data = abc_model.wv.most_similar('hell')
print(abc_data)

RM_voc = list(RM_model.wv.vocab)
print("Most similar to hell according to RM_model: ")
RM_data = RM_model.wv.most_similar('hell')
print(RM_data)

############ Similarity tests  ##############################
similar_words = abc_model.wv.most_similar('god')	
print(similar_words)
print()

similar_words = RM_model.wv.most_similar('god')	
print(similar_words)
print()
#############
dissimlar_words = abc_model.wv.doesnt_match('god good hell'.split())
print(dissimlar_words)
print()
dissimlar_words = RM_model.wv.doesnt_match('god good hell'.split())
print(dissimlar_words)
print()

#################
similarity_two_words = abc_model.wv.similarity('human','rat')
print("Please provide the similarity between these two words:")
print(similarity_two_words)
print()

similarity_two_words = RM_model.wv.similarity('human', 'rat')
print("Please provide the similarity between these two words:")
print(similarity_two_words)
print()

##########
similar = abc_model.wv.similar_by_word('kind')
print(similar)
print()
similar = RM_model.wv.similar_by_word('kind')
print(similar)
print()


###### Loading a model ################################
#model = Word2Vec.load('model.bin')
#vocab = list(model.wv.vocab)
#print()
#######################################################



############################################################
#########################  PCA #############################

######## passing vectors into df ######################
RM_model.wv.vocab
RM_model.wv.vectors.shape
RM_model.corpus_total_words

DF_RM_vectors = pd.DataFrame(data = RM_model.wv.vectors[0:,0:],
                index=[i for i in range(RM_model.wv.vectors.shape[0])],
                columns=['f'+str(i) for i in range(RM_model.wv.vectors.shape[1])])
DF_RM_vectors.tail()
 

abc_model.wv.vocab
abc_model.wv.vectors.shape
abc_model.corpus_total_words

DF_abc_vectors = pd.DataFrame(data = abc_model.wv.vectors[0:,0:],
                index=[i for i in range(abc_model.wv.vectors.shape[0])],
                columns=['f'+str(i) for i in range(abc_model.wv.vectors.shape[1])])
DF_abc_vectors.tail()
############################################################

########## Re-scaling data ######################################
### RM
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(RM_model.wv.vectors)
# Apply transform to both the training set and the test set.
scaled_RM_vectors = scaler.transform(RM_model.wv.vectors)
scaled_RM_vectors  = scaler.transform(RM_model.wv.vectors)

RM_pca = PCA(n_components=2)
RM_pca.fit(RM_model.wv.vectors)
RM_pca.n_components_

RM_pca.explained_variance_
scaled_RM_vectors = RM_pca.transform(RM_model.wv.vectors)

### abc
abc_scaler = StandardScaler()
# Fit on training set only.
abc_scaler.fit(abc_model.wv.vectors)
# Apply transform to both the training set and the test set.
scaled_abc_vectors = abc_scaler.transform(abc_model.wv.vectors)
scaled_RM_vectors  = abc_scaler.transform(abc_model.wv.vectors)

abc_pca = PCA(n_components=2)
abc_pca.fit(abc_model.wv.vectors)

abc_pca.explained_variance_
scaled_abc_vectors = abc_pca.transform(abc_model.wv.vectors)

#### plotting RM vocabulary  ###############
plt.figure(figsize=(20,20))
plt.scatter(scaled_RM_vectors[:, 0], scaled_RM_vectors[:, 1])
plt.xlabel("$PCA_1$", fontsize=14)
plt.ylabel("$PCA_2$", fontsize=14)
plt.axis('equal')
plt.show()
################## clustering ############
### specifying some colors
number = 100 # of colors/clusters
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

kmeans = KMeans(number)
kmeans.fit(scaled_RM_vectors)
RM_clusters = kmeans.predict(scaled_RM_vectors)


plt.figure(figsize=(20,20))
for i, color in enumerate(colors, start=1):
    plt.scatter(scaled_RM_vectors[RM_clusters==i,0], 
      scaled_RM_vectors[RM_clusters==i,1], label="$k$ Cluster " + str(i),color = color)
plt.legend(fontsize=14)
plt.show()


##### plotting abc vocabulary  ###############
plt.figure(figsize=(20,20))
plt.scatter(scaled_abc_vectors[:, 0], scaled_abc_vectors[:, 1])
plt.xlabel("$PCA_1$", fontsize=14)
plt.ylabel("$PCA_2$", fontsize=14)
plt.axis('equal')
plt.show()
################## clustering ############
### specifying some colors
number = 50 # of colors/clusters
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

kmeans = KMeans(number)
kmeans.fit(scaled_abc_vectors)
abc_clusters = kmeans.predict(scaled_abc_vectors)


plt.figure(figsize=(20,20))
for i, color in enumerate(colors, start=1):
    plt.scatter(scaled_abc_vectors[abc_clusters==i,0], 
      scaled_abc_vectors[abc_clusters==i,1], label="$k$ Cluster " + str(i),color = color)
plt.legend(fontsize=14)
plt.show()



##################################################################
###### Representing songs as vectors ############################
#dropping words NOT in vocabulary

count = 0
for index, row in Pro_Lyrics_train.iterrows():
    count = 0
    for w in row['words']:
        if w in RM_model.wv.vocab:
            print(row['words'][count], "row: ", count)            
            count = count+1           

Xx = np.zeros(0)
if type([1,2,3,]) is list:
        print("yes")

###present = np.empty(len(Pro_Lyrics_train), dtype=list)
        
####### checks for all words that were (not) vectorized ###
## stores 0 or 1 in "in vocabulary" column
turtle = []
present = [turtle for i in range(len(Pro_Lyrics_train))]
Pro_Lyrics_train['in_vocabulary'] = present
Pro_Lyrics_train.tail(5)


Zv = [np.zeros(150) for i in range(len(Pro_Lyrics_train))]
Pro_Lyrics_train['s_vector'] = Zv

Pro_Lyrics_train.sample(3)

### Looking for words in RM_model.vocab else, to then delete words
count = 0
for index, row in Pro_Lyrics_train.iterrows():
    count +=1
    for w in row['words']:
        if w in RM_model.wv.vocab:
            #print(RM_model.wv.word_vec(w))
            #print('smt')
            #print(row['in_vocabulary'])
            row['in_vocabulary'] = row['in_vocabulary'] + [1] #means w is part of RM_vocab
        else:
            row['in_vocabulary'] = row['in_vocabulary'] + [0] #means w is not part of RM_vocab
    if(count % 10000 == 0):
        print(count)
Pro_Lyrics_train.tail(3)

print(len(Pro_Lyrics_train['in_vocabulary']))
##################################




############### representing a song by the average of its words  ##################################
# Computing the average of vectorized words in "in vocabulary"
past_iter = np.zeros(150)
count = 0
num = 0
dummy = 0
Pro_Lyrics_train_song_mean = []
for index, row in Pro_Lyrics_train.iterrows():
    num = 0
    temp = np.zeros(150)
    #for i in row['in_vocabulary']:
    #    if(i == 1):
    #        #print(row['words'][i])#=w
    #        temp = temp + RM_model.wv.word_vec(row['words'][i])
    #        num += 1     
    dummy = 0    
    for w in row['words']:
        if(row['in_vocabulary'][dummy] == 1):
            #print("w: ", RM_model.wv.word_vec(w))
            #print(row['in_vocabulary'][dummy])
            #print(w)#row['words'][i])#=w
            #print("dummy: ", dummy)
            temp = temp + RM_model.wv.word_vec(w)
            num += 1
            #print("temp: ",temp)
            #print("num: ", num)
        #else:
            #print("problematic word!!")
            #print(row['in_vocabulary'][dummy])
            #print(w)#row['words'][i])#=w
            #print(dummy)    
        dummy += 1
    if(num == 0):    
        print("show me the money!")
        row['s_vector'] = np.zeros(150)        
    else:
        num = num*1.0
        #print("I know kung fu!")
        #print("s_vec: ", row['s_vector'])
        #if(count % 1000 == 0):
        #     print("s_vector: ", row['s_vector'])
        #     if ((past_iter==row['s_vector']).all()):
        #         print("No changes", )
        #     past_iter = row['s_vector']
        row['s_vector'] = temp/num
        #print("s_vec: ", row['s_vector'])
    count += 1
    if(count % 10000 == 0):
        print(count)
        print(num)
         
Pro_Lyrics_train.sample(30)
#######################################


print(Pro_Lyrics_train['words'][21564][0])
del Pro_Lyrics_train['words'][21564][0]
##################################################################



######### DecisionTree Classifier  ###############
Pro_Lyrics_train_clf = DecisionTreeClassifier(random_state = 440)
# Fit the model
Pro_Lyrics_train_clf.fit(list(Pro_Lyrics_train.s_vector), type_train.type)
###########

# Plot the fitted tree
plt.figure(figsize = (20,20))
fig = Pro_Lyrics_train_clf.fit(list(Pro_Lyrics_train.s_vector), type_train.type)
tree.plot_tree(fig,filled = True)
plt.show()
##################        

###present = np.empty(len(Pro_Lyrics_train), dtype=list)


####### checks for all words that were (not) vectorized ###
## stores 0 or 1 in "in vocabulary" column
turtle = []
present = [turtle for i in range(len(Pro_Lyrics_test))]
Pro_Lyrics_test['in_vocabulary'] = present
Pro_Lyrics_test.tail(5)


Zv = [np.zeros(150) for i in range(len(Pro_Lyrics_test))]
Pro_Lyrics_test['s_vector'] = Zv

Pro_Lyrics_test.sample(3)

### Looking for words in RM_model.vocab else, to then delete words
count = 0
for index, row in Pro_Lyrics_test.iterrows():
    count +=1
    for w in row['words']:
        if w in RM_model.wv.vocab:
            #print(RM_model.wv.word_vec(w))
            #print('smt')
            #print(row['in_vocabulary'])
            row['in_vocabulary'] = row['in_vocabulary'] + [1] #means w is part of RM_vocab
        else:
            row['in_vocabulary'] = row['in_vocabulary'] + [0] #means w is not part of RM_vocab
    if(count % 10000 == 0):
        print(count)
Pro_Lyrics_test.sample(5)

print(len(Pro_Lyrics_test['in_vocabulary']))
################


############### representing a song by the average of its words  ##################################
# Computing the average of vectorized words in "in vocabulary"
past_iter = np.zeros(150)
count = 0
num = 0
dummy = 0
Pro_Lyrics_test_song_mean = []
for index, row in Pro_Lyrics_test.iterrows():
    num = 0
    temp = np.zeros(150)
    dummy = 0    
    for w in row['words']:
        if(row['in_vocabulary'][dummy] == 1):
            temp = temp + RM_model.wv.word_vec(w)
            num += 1
        dummy += 1
    if(num == 0):    
        print("show me the money!")
        row['s_vector'] = np.zeros(150)        
    else:
        num = num*1.0
        row['s_vector'] = temp/num
    count += 1
    if(count % 1000 == 0):
        print(count)
        print(num)
        
Pro_Lyrics_test.sample(10)        
#######################################################       

######################### Making predictions with Tree ########
test_predictions_word2vec = Pro_Lyrics_train_clf.predict(list(Pro_Lyrics_test['s_vector']))

print(classification_report(type_test['type'], test_predictions_word2vec))
###############################################################