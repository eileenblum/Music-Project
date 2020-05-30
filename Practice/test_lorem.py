from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

import gzip
import gensim
import logging
import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data_file="reviews_data.txt.gz"

with gzip.open ('reviews_data.txt.gz', 'rb') as f:
    for i,line in enumerate (f):
        print(line)
        break

def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    
    logging.info("reading file {0}...this may take a while".format(input_file))
    
    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line)

# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list (read_input (data_file))
logging.info ("Done reading data file")    





"""
#Training a model
word2vec.word2phrase('lorem_Ipsum.txt', 'lorem_Ipsum_phrases', verbose=True)


word2vec.word2vec('lorem_Ipsum_phrases', 'lorem_Ipsum.bin', 
                  size=100, binary=True, verbose=True)

word2vec.word2clusters('lorem_Ipsum.txt', 'lorem_Ipsum_clusters.txt',
                       100, verbose=True)

#Predictions
model = word2vec.load('lorem_Ipsum.bin')

model.vocab
"""
