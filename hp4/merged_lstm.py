# 3 merged LSTM for all   
import numpy

from keras.models import Sequential
from keras.layers import Dense,LSTM, Merge, InputLayer
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding

from keras.utils.visualize_util import plot

# fix random seed for reproducibility
numpy.random.seed(7)
# keep the top n words, zero the rest
top_item = 2488
# truncate and pad input sequences
max_review_length = 130


# create the model
embedding_vecor_length = 32
textInput = Embedding(top_item, embedding_vecor_length, input_length=max_review_length, dropout=0.2, name="EMBEDED_TRANSCRIPT")

modelText = Sequential(name="TRANSCRIPT")
modelText.add(textInput)
modelText.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
modelText.add(MaxPooling1D(pool_length=2))
modelText.add(LSTM(100, dropout_W=0.1, dropout_U=0.1))
modelText.add(Dense(10, activation='relu'))
modelText.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


plot(modelText, to_file='plots/modelText.png', show_layer_names=True, show_shapes=True)

# Start Audio input

audioInput = InputLayer(input_shape=(1,100,), name="INPUT_AUDIO")
modelAudio = Sequential(name="AUDIO")
modelAudio.add(audioInput)
modelAudio.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
modelAudio.add(MaxPooling1D(pool_length=2))
modelAudio.add(LSTM(100, dropout_W=0.1, dropout_U=0.1))
modelAudio.add(Dense(10, activation='relu'))
modelAudio.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

plot(modelAudio, to_file='plots/modelAudio.png', show_layer_names=True, show_shapes=True)


# Start Video input

videoInput = InputLayer(input_shape=(1,200,), name="INPUT_VIDEO")
modelVideo = Sequential(name="FACIAL")
modelVideo.add(videoInput)
modelVideo.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
modelVideo.add(MaxPooling1D(pool_length=2))
modelVideo.add(LSTM(100, dropout_W=0.1, dropout_U=0.1))
modelVideo.add(Dense(10, activation='relu'))
modelVideo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

plot(modelVideo, to_file='plots/modelVideo.png', show_layer_names=True, show_shapes=True)

mergeModel = Sequential()
mergeModel.add(Merge([modelText,modelAudio, modelVideo], mode='concat', concat_axis=1))
mergeModel.add(Dense(256, activation='relu'))
mergeModel.add(Dense(1, activation='sigmoid'))


plot(mergeModel, to_file='plots/mergeModel.png', show_layer_names=True, show_shapes=True)

