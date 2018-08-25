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
from keras.layers.core import Dense , Dropout , Activation
from keras.layers import Embedding , LSTM, BatchNormalization, SpatialDropout1D
from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight

# size of the word embeddings
embeddings_dim = 100

# maximum number of words to consider in the representations
max_features = 20000

# maximum length of a sentence
max_sent_len = 50

# percentage of the data used for model training
percent = 0.80

# number of classes
num_classes = 5

path_to_glove_embed = 'glove.6B.100d.txt'

embeddings = dict( )

print ("Lettura file...")
'''
datasets:
ENG: test-data-rated.txt
ITA: newdataset.txt
'''

#ITA-VERSION
'''
data = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("newdataset.txt", encoding="utf8"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
random.shuffle( data )
train_size = int(len(data))

train_texts = [ txt.lower() for ( txt, label ) in data[0:train_size] ]
train_labels = [ label for ( txt , label ) in data[0:train_size] ]

datatest = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("testset.txt", encoding="utf8"), delimiter='\t', quoting=csv.QUOTE_NONE) ]

test_size = int(len(datatest))

test_texts = [ txt.lower() for ( txt, label ) in datatest[0:test_size] ]
test_labels = [ label for ( txt , label ) in datatest[0:test_size] ]
'''

#ENG-VERSION

data = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("test-data-rated.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
random.shuffle( data )
train_size = int(len(data) * percent)

train_texts = [ txt.lower() for ( txt, label ) in data[0:train_size] ]
train_labels = [ label for ( txt , label ) in data[0:train_size] ]

test_texts = [ txt.lower() for ( txt, label ) in data[train_size:-1] ]
test_labels = [ label for ( txt , label ) in data[train_size:-1] ]



embeddings =  gensim.models.Word2Vec(train_texts, min_count=1, size=300)

tokenizer = Tokenizer(num_words=max_features,lower=True)
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
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

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_labels),
                                                  train_labels)
class_weights_dict = dict(zip(le.transform(list(le.classes_)),
                              class_weights))

train_labels = le.transform( train_labels )
test_labels = le.transform( test_labels )
print("Classi considerate: " + repr( le.classes_ ))

train_labels = to_categorical(train_labels)

np.random.seed(0)

"""pre-trained Glove"""
embeddings_index = dict()

#ita_version
#http://hlt.isti.cnr.it/wordembeddings/glove_wiki_window10_size300_iteration50.tar.gz
#embeddings_index = Word2Vec.load('glove_WIKI') # glove model
#embeddings_index.wv.save_word2vec_format('word2vec_ita.txt', binary=False)

#f = open("word2vec_ita.txt", encoding="utf8")

#eng_version
#http://nlp.stanford.edu/data/glove.6B.zip
f = open("glove.6B.300d.txt", encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=50,trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(64,activation='tanh', dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

adam=keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


print("---CLASS WEIGHTS---")
print(class_weights_dict)

model.fit(train_sequences, train_labels , epochs=10, batch_size=32, class_weight=class_weights_dict)

results = model.predict_classes( test_sequences )

count = 0
f = open("results.txt", "w+", newline="\n")
for i in range(len(results)):
    if test_labels[i] == results[i]:
        count = count+1;
        
    f.write("stringa: " + str(test_texts[i]) + "\n")
    f.write("res: " + str(test_labels[i]) + "\n")
    f.write("pred: " + str(results[i]) + "\n")

f.write("numero corretti: " + str(count) + " su " + str(len(test_labels)) + "\n")


f.close() 
