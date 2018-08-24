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
from keras.layers import Embedding , LSTM, BatchNormalization, SpatialDropout1D
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

print ("")
print ("Reading pre-trained word embeddings...")
path_to_glove_embed = '/home/upasana/Documents/ubunt/Glove/glove.twitter.27B.100d.txt'


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
print("Classes that are considered in the problem : " + repr( le.classes_ ))

train_labels = to_categorical(train_labels)

print("-----TEST-SEQUENCE-SHAPE-----")
print(test_sequences.shape)

print ("Method = Stack of two LSTMs")
np.random.seed(0)

""" Load pre-trained Glove Emedding from Stanford CoreNLP package"""
embeddings_index = dict()
f = open(path_to_glove_embed, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=50,trainable=False))
# model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_matrix]))
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


model.fit(train_sequences, train_labels , epochs=3, batch_size=32, class_weight=class_weights_dict)

results = model.predict_classes( test_sequences )

print(results)

for i in range(len(results)):
    print(test_texts[i])
    print('res: ', test_labels[i])
    print('pred: ', results[i])
