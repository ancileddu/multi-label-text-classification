import numpy as np
import csv
import keras
import sklearn
import gensim
import random
import scipy
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense , Dropout , Activation  , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from sklearn.naive_bayes import MultinomialNB
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from keras.utils.np_utils import to_categorical
from gensim.test.utils import common_texts
from gensim.models import FastText
from keras.layers import Conv2D, MaxPooling2D

# size of the word embeddings
embeddings_dim = 300

# maximum number of words to consider in the representations
max_features = 30000

# maximum length of a sentence
max_sent_len = 50

# percentage of the data used for model training
percent = 0.75

# number of classes
num_classes = 5

print ("")
print ("Reading pre-trained word embeddings...")

#find the file on https://github.com/mmihaltz/word2vec-GoogleNews-vectors
embeddings = dict( )
#embeddings = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz" , binary=True)

#or with fasttext
#embeddings = FastText(common_texts, size=4, window=3, min_count=1, iter=10)


print ("Reading text data for classification and building representations...")
data = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("test-data-rated.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
random.shuffle( data )
train_size = int(len(data) * percent)

train_texts = [ txt.lower() for ( txt, label ) in data[0:train_size] ]
test_texts = [ txt.lower() for ( txt, label ) in data[train_size:-1] ]

embeddings =  gensim.models.Word2Vec(train_texts, min_count=1, size=300)

train_labels = [ label for ( txt , label ) in data[0:train_size] ]
test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
num_classes = len( set( train_labels + test_labels ) )
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tokenizer.fit_on_texts(train_texts)
train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
train_matrix = tokenizer.texts_to_matrix( train_texts )
test_matrix = tokenizer.texts_to_matrix( test_texts )
embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
for word,index in tokenizer.word_index.items():
  if index < max_features:
    try: embedding_weights[index,:] = embeddings[word]
    except: embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )
    
le = preprocessing.LabelEncoder( )
le.fit( train_labels + test_labels )
train_labels = le.transform( train_labels )
test_labels = le.transform( test_labels )
print("Classes that are considered in the problem : " + repr( le.classes_ ))

train_labels = to_categorical(train_labels)

print("-----TEST-SEQUENCE-SHAPE-----")
print(test_sequences.shape)

print ("Method = Stack of two LSTMs")
np.random.seed(0)


model = Sequential()

model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights] ))
model.add(Dropout(0.25))
model.add(LSTM(output_dim=embeddings_dim , activation='tanh', dropout=0.2, recurrent_dropout=0.2, inner_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(activation='tanh', units=embeddings_dim, dropout=0.2, recurrent_dropout=0.2, recurrent_activation='hard_sigmoid', return_sequences=False))
model.add(Dropout(0.25))

model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

adam=keras.optimizers.Adam(lr=0.04)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


model.fit(train_sequences, train_labels , epochs=3, batch_size=32)

results = model.predict_classes( test_sequences )

print(results)

for i in range(len(results)):
    print(test_texts[i])
    print('res: ', test_labels[i])
    print('pred: ', results[i])