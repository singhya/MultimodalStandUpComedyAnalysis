import json
# LSTM for sequence classification in the IMDB dataset
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy

a_set = set()
with open('words_per_segment.json') as f:
    lst = json.load(f)
    for row in lst:
        a_set = a_set.union(set(row))
    print(lst)

my_list = list(a_set)
dict = {}
for index,item in enumerate(my_list):
    dict[item] = index
x = []
max_length = 0
for index,row in enumerate(lst):
    int_row = []
    for word in row:
        int_row.append(dict[word])
    print (str(index)+" "+str(len(int_row)))
    x.append(int_row)
    max_length = max(max_length, len(int_row))
y = []
import csv
with open('segment_annotation_and_list_of_words.csv', 'rb') as csvfile:
    content = csv.reader(csvfile, delimiter=' ', quotechar='|')
    next(content)
    for row in content:
         y.append(row[0].split(",")[3])

size = len(x)
x = numpy.array(x)
Y = pandas.read_csv('segment_annotation.csv', names=['video','laughter_start','laughter_end','laughter_value','start_segment_s','end_segment_s','start_segment_frame','end_segment_frame'])
Y = numpy.array(Y['laughter_value'])
Y = numpy.delete(Y, 0)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

separator = int(.8*size)
X_train, X_test = x[:separator], x[separator:]
y_train, y_test = dummy_y[:separator], dummy_y[separator:]

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
nb_classes = 3
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_split=0.25, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
y_predicted = model.predict(X_test, batch_size=32, verbose=0)
numpy.savetxt('LSTMTranscript_output.csv', y_predicted, delimiter=',')
numpy.savetxt('Test.csv', y_test, delimiter=',')
print("Accuracy: %.2f%%" % (scores[1]*100))