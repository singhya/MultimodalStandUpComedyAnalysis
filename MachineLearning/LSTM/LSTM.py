import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import numpy

frames = 15
Y = pandas.read_csv('segment_annotation.csv',
                    names=['video', 'laughter_start', 'laughter_end', 'laughter_value', 'start_segment_s',
                           'end_segment_s', 'start_segment_frame', 'end_segment_frame'])
'''audio_cols = ['frameTime','pcm_fftMag_mfcc[0]', 'pcm_fftMag_mfcc[1]', 'pcm_fftMag_mfcc[2]','pcm_fftMag_mfcc[3]', 'pcm_fftMag_mfcc[4]', 'pcm_fftMag_mfcc[5]','pcm_fftMag_mfcc[6]', 'pcm_fftMag_mfcc[7]', 'pcm_fftMag_mfcc[8]','pcm_fftMag_mfcc[9]', 'pcm_fftMag_mfcc[10]', 'pcm_fftMag_mfcc[11]','pcm_fftMag_mfcc[12]', 'pcm_fftMag_mfcc_de[0]', 'pcm_fftMag_mfcc_de[1]','pcm_fftMag_mfcc_de[5]', 'pcm_fftMag_mfcc_de[6]', 'pcm_fftMag_mfcc_de[7]','pcm_fftMag_mfcc_de[2]', 'pcm_fftMag_mfcc_de[3]', 'pcm_fftMag_mfcc_de[4]','pcm_fftMag_mfcc_de[8]', 'pcm_fftMag_mfcc_de[9]', 'pcm_fftMag_mfcc_de[10]','pcm_fftMag_mfcc_de[11]', 'pcm_fftMag_mfcc_de[12]', 'pcm_fftMag_mfcc_de_de[0]','pcm_fftMag_mfcc_de_de[1]', 'pcm_fftMag_mfcc_de_de[2]','pcm_fftMag_mfcc_de_de[3]', 'pcm_fftMag_mfcc_de_de[4]','pcm_fftMag_mfcc_de_de[5]', 'pcm_fftMag_mfcc_de_de[6]','pcm_fftMag_mfcc_de_de[7]', 'pcm_fftMag_mfcc_de_de[8]','pcm_fftMag_mfcc_de_de[9]', 'pcm_fftMag_mfcc_de_de[10]','pcm_fftMag_mfcc_de_de[11]', 'pcm_fftMag_mfcc_de_de[12]', 'chroma[0]','chroma[1]', 'chroma[2]', 'chroma[3]', 'chroma[4]', 'chroma[5]', 'chroma[6]','chroma[7]', 'chroma[8]', 'chroma[9]', 'chroma[10]', 'chroma[11]','chroma[0].1', 'chroma[1].1', 'chroma[2].1', 'chroma[3].1', 'chroma[4].1','chroma[5].1', 'chroma[6].1', 'chroma[7].1', 'chroma[8].1', 'chroma[9].1','chroma[10].1', 'chroma[11].1', 'pcm_LOGenergy', 'voiceProb_sma', 'F0_sma','pcm_loudness_sma', 'F0final_sma', 'voicingFinalUnclipped_sma','pcm_loudness_sma.1']
video_cols = ['frame','timestamp','confidence','success','gaze_0_x','gaze_0_y','gaze_0_z','gaze_1_x','gaze_1_y','gaze_2_z','pose_Tx','pose_Ty','pose_Tz','pose_Rx','pose_Ry','pose_Rz','x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','x_9','x_10','x_11','x_12','x_13','x_14','x_15','x_16','x_17','x_18','x_19','x_20','x_21','x_22','x_23','x_24','x_25','x_26','x_27','x_28','x_29','x_30','x_31','x_32','x_33','x_34','x_35','x_36','x_37','x_38','x_39','x_40','x_41','x_42','x_43','x_44','x_45','x_46','x_47','x_48','x_49','x_50','x_51','x_52','x_53','x_54','x_55','x_56','x_57','x_58','x_59','x_60','x_61','x_62','x_63','x_64','x_65','x_66','x_67','y_0','y_1','y_2','y_3','y_4','y_5','y_6','y_7','y_8','y_9','y_10','y_11','y_12','y_13','y_14','y_15','y_16','y_17','y_18','y_19','y_20','y_21','y_22','y_23','y_24','y_25','y_26','y_27','y_28','y_29','y_30','y_31','y_32','y_33','y_34','y_35','y_36','y_37','y_38','y_39','y_40','y_41','y_42','y_43','y_44','y_45','y_46','y_47','y_48','y_49','y_50','y_51','y_52','y_53','y_54','y_55','y_56','y_57','y_58','y_59','y_60','y_61','y_62','y_63','y_64','y_65','y_66','y_67','p_scale','p_rx','p_ry','p_rz','p_tx','p_ty','p_0','p_1','p_2','p_3','p_4','p_5','p_6','p_7','p_8','p_9','p_10','p_11','p_12','p_13','p_14','p_15','p_16','p_17','p_18','p_19','p_20','p_21','p_22','p_23','p_24','p_25','p_26','p_27','p_28','p_29','p_30','p_31','p_32','p_33','AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']
'''
cols = ['frameTime', 'pcm_fftMag_mfcc[0]', 'pcm_fftMag_mfcc[1]', 'pcm_fftMag_mfcc[2]', 'pcm_fftMag_mfcc[3]',
        'pcm_fftMag_mfcc[4]', 'pcm_fftMag_mfcc[5]', 'pcm_fftMag_mfcc[6]', 'pcm_fftMag_mfcc[7]', 'pcm_fftMag_mfcc[8]',
        'pcm_fftMag_mfcc[9]', 'pcm_fftMag_mfcc[10]', 'pcm_fftMag_mfcc[11]', 'pcm_fftMag_mfcc[12]',
        'pcm_fftMag_mfcc_de[0]', 'pcm_fftMag_mfcc_de[1]', 'pcm_fftMag_mfcc_de[5]', 'pcm_fftMag_mfcc_de[6]',
        'pcm_fftMag_mfcc_de[7]', 'pcm_fftMag_mfcc_de[2]', 'pcm_fftMag_mfcc_de[3]', 'pcm_fftMag_mfcc_de[4]',
        'pcm_fftMag_mfcc_de[8]', 'pcm_fftMag_mfcc_de[9]', 'pcm_fftMag_mfcc_de[10]', 'pcm_fftMag_mfcc_de[11]',
        'pcm_fftMag_mfcc_de[12]', 'pcm_fftMag_mfcc_de_de[0]', 'pcm_fftMag_mfcc_de_de[1]', 'pcm_fftMag_mfcc_de_de[2]',
        'pcm_fftMag_mfcc_de_de[3]', 'pcm_fftMag_mfcc_de_de[4]', 'pcm_fftMag_mfcc_de_de[5]', 'pcm_fftMag_mfcc_de_de[6]',
        'pcm_fftMag_mfcc_de_de[7]', 'pcm_fftMag_mfcc_de_de[8]', 'pcm_fftMag_mfcc_de_de[9]', 'pcm_fftMag_mfcc_de_de[10]',
        'pcm_fftMag_mfcc_de_de[11]', 'pcm_fftMag_mfcc_de_de[12]', 'chroma[0]', 'chroma[1]', 'chroma[2]', 'chroma[3]',
        'chroma[4]', 'chroma[5]', 'chroma[6]', 'chroma[7]', 'chroma[8]', 'chroma[9]', 'chroma[10]', 'chroma[11]',
        'chroma[0].1', 'chroma[1].1', 'chroma[2].1', 'chroma[3].1', 'chroma[4].1', 'chroma[5].1', 'chroma[6].1',
        'chroma[7].1', 'chroma[8].1', 'chroma[9].1', 'chroma[10].1', 'chroma[11].1', 'pcm_LOGenergy', 'voiceProb_sma',
        'F0_sma', 'pcm_loudness_sma', 'F0final_sma', 'voicingFinalUnclipped_sma', 'pcm_loudness_sma.1', 'frame',
        'timestamp', 'confidence', 'success', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_2_z',
        'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5',
        'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19',
        'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33',
        'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47',
        'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61',
        'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8',
        'y_9', 'y_10', 'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20', 'y_21', 'y_22',
        'y_23', 'y_24', 'y_25', 'y_26', 'y_27', 'y_28', 'y_29', 'y_30', 'y_31', 'y_32', 'y_33', 'y_34', 'y_35', 'y_36',
        'y_37', 'y_38', 'y_39', 'y_40', 'y_41', 'y_42', 'y_43', 'y_44', 'y_45', 'y_46', 'y_47', 'y_48', 'y_49', 'y_50',
        'y_51', 'y_52', 'y_53', 'y_54', 'y_55', 'y_56', 'y_57', 'y_58', 'y_59', 'y_60', 'y_61', 'y_62', 'y_63', 'y_64',
        'y_65', 'y_66', 'y_67', 'p_scale', 'p_rx', 'p_ry', 'p_rz', 'p_tx', 'p_ty', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4',
        'p_5', 'p_6', 'p_7', 'p_8', 'p_9', 'p_10', 'p_11', 'p_12', 'p_13', 'p_14', 'p_15', 'p_16', 'p_17', 'p_18',
        'p_19', 'p_20', 'p_21', 'p_22', 'p_23', 'p_24', 'p_25', 'p_26', 'p_27', 'p_28', 'p_29', 'p_30', 'p_31', 'p_32',
        'p_33', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
        'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c',
        'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c',
        'AU26_c', 'AU28_c', 'AU45_c']
A = numpy.array(numpy.repeat(range(0, frames), len(cols)))
B = numpy.array(cols * frames)
mean = pandas.read_csv('std_data.csv', names=pandas.MultiIndex.from_tuples(zip(A, B), names=['first', 'second']))
df = mean.loc[:, mean.columns.get_level_values(1)
    .isin(['pcm_LOGenergy', 'F0_sma', 'pcm_loudness_sma', 'pcm_fftMag_mfcc[0]', 'pcm_fftMag_mfcc[1]', 'chroma[0]',
           'chroma[1]', 'chroma[2]', 'chroma[3]', 'AU07_c', 'AU14_c', 'AU23_c', 'pose_Rx', 'pose_Ry', 'gaze_0_y',
           'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
           'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'])]

df.convert_objects(convert_numeric=True)
X = numpy.array(df)


def runLSTM(X_train, X_test, y_train, y_test,b):
    # fix random seed for reproducibility
    numpy.random.seed(7)
    model = Sequential()

    'exception_verbosity = high'
    batch_size = b
    hidden_units = 71
    nb_classes = 3

    model.add(LSTM(batch_size, input_shape=X_train.shape[1:]))
    model.add(Dense(nb_classes, activation='softplus'))
    # model.add(BatchNormalization((batch_size),weights=None))
    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
    # model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, nb_epoch=30, validation_split=0.25, batch_size=batch_size, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=2)
    y_predicted = model.predict(X_test, batch_size=32, verbose=0)
    numpy.savetxt('LSTM_output.csv', y_predicted, delimiter=',')
    numpy.savetxt('Test.csv', y_test, delimiter=',')
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    return scores[1]

X[2:] = X[2:].astype(numpy.float)
X = X.reshape((len(X), frames, len(X[0]) / frames))
X = numpy.delete(X, [0, 1], axis=0)
size = len(X)

Y = numpy.array(Y['laughter_value'])
Y = numpy.delete(Y, 0)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
X = X.astype(numpy.float)
Y = Y.astype(numpy.float)

#separator = int(.8 * size)
#X_train, X_test = X[:separator], X[separator:]
#y_train, y_test = dummy_y[:separator], dummy_y[separator:]

results = {}
test_size = int(.2 * size)
for b in range(25,175):
    result = []
    for i in range(0,5):
        X_train = X[:i*test_size]
        X_test = X[i * test_size:(i * test_size) + test_size]
        X_train = numpy.vstack((X_train, X[(i*test_size)+test_size:]))
        y_train = dummy_y[:i*test_size]
        y_test = dummy_y[i * test_size:(i * test_size) + test_size]
        y_train = numpy.vstack((y_train, dummy_y[(i * test_size) + test_size:]))
        result.append(runLSTM(X_train, X_test,y_train,y_test,b))
    results[b]=result

for i in results:
    print(str(i) + " : ")
    print(results[i])