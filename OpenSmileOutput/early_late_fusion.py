import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import linear_model


def get_features(columns):
    segmentation_annotation_labels = ['video', 'laughter_start', 'laughter_end', 'laughter_value',
       'start_segment_s', 'end_segment_s', 'start_segment_frame',
       'end_segment_frame','video_num']
    
    return sorted(list(set(list(columns)) - set(segmentation_annotation_labels)))



def late_fusion(audio_input_df,
                video_input_df,
                transcript_input_df,
                audio_features,
                video_features,
                transcript_features):
    #taking video num from audio data frame to divide video num for 5 experiments.     
    video_num_df = pd.DataFrame(list(audio_input_df.video_num.unique()))
    kf = KFold(n_splits=5)
    kf.get_n_splits(video_num_df)

    experiment = 0 
    for train_index, test_index in kf.split(video_num_df):
        print('Experiment : ', experiment)
        experiment = experiment + 1 

        #early fusion 

        df0 = audio_input_df
        df1 = video_input_df[video_features]#video_std_df[video_std_features]
        df2 = transcript_input_df[transcript_features]#transcript_input_df[transcript_features]
        early_fusion_frame = [df0,df1,df2]

        df0.index = df1.index 
        early_fusion_df = pd.concat(early_fusion_frame, axis=1)
        early_fusion_features = audio_features + video_features + transcript_features
        
        [early_fusion_prediction,early_fusion_accuracy] = get_prediction(early_fusion_df,
                                                                        early_fusion_features,
                                                                        train_index,
                                                                        test_index)

        print("Early fusion : ",early_fusion_accuracy)
        
        #audio predition 
        
        [audio_prediction,audio_accuracy] = get_prediction(audio_input_df,
                                                           audio_features,
                                                           train_index,
                                                           test_index)
        
        print('Audio: ',audio_accuracy)
        
        #video prediction
       
        [video_predition, video_accuracy] = get_prediction(video_input_df,
                                                           video_features,
                                                           train_index,
                                                           test_index)
        print('Video: ',video_accuracy)
        
        #transcript prediction
        
        [transcript_prediction,transcript_accuracy] = get_prediction(transcript_input_df,
                                                                     transcript_features,
                                                                     train_index,
                                                                     test_index)
        print('Transcript: ',transcript_accuracy)
        
        
        late_fusion_list = []
        for i in range(len(audio_prediction)):
            late_fusion_list.append([
                    audio_prediction[i],
                    video_predition[i],
                    transcript_prediction[i]
                    ])
        
        accuracies = [audio_accuracy,video_accuracy,transcript_accuracy]
        
        max_accuracy = max(accuracies)
        max_index = accuracies.index(max_accuracy)
        
        late_fusion_result = []
        
        for segment_prediction in late_fusion_list:
            result = max_index
            if len(segment_prediction) != len(set(segment_prediction)):
                result = max(set(segment_prediction), key=segment_prediction.count)
            
            late_fusion_result.append(result)
            
        
        #print(late_fusion_result)
        	
	#any other input data frame can be used. i've choosen audio_input_df. This is mainly to get laugther values. for test data.  
        test_data = []
        for video_num in test_index: 
            test_data.append(audio_input_df[audio_input_df.video_num == video_num])
        test_data = pd.concat(test_data)
        
        test_y = list(test_data.laughter_value)
        
        late_fusion_accuracy = accuracy_score(late_fusion_result,test_y) 
        
        print('late fusion accuracy : ', late_fusion_accuracy)
        
        
        print('+'*20)
        
def get_prediction(input_df,features,train_index,test_index):
    
    #print('shape of the input dataframe : ', input_df.shape)
    #print('number of features : ', len(features))
    train_data = []

    for video_num in train_index: 
        train_data.append(input_df[input_df.video_num == video_num])

    train_data = pd.concat(train_data)

    #test data 

    test_data = []

    for video_num in test_index: 
        test_data.append(input_df[input_df.video_num == video_num])

    test_data = pd.concat(test_data)


    #validation process 

    video_num_df1 = pd.DataFrame(list(train_data.video_num.unique()))
    kf1 = KFold(n_splits=4)
    kf1.get_n_splits(video_num_df1)

    fold = 0
    for tr_index,val_index in kf1.split(video_num_df1):
        #print('Fold : ' ,fold)
        fold = fold + 1 

        #validation train data
        tr_data = []

        for video_num in tr_index:
            tr_data.append(input_df[input_df.video_num == video_num])

        tr_data = pd.concat(tr_data)


        #validation data 
        val_data = []

        for video_num in val_index:
            val_data.append(input_df[input_df.video_num == video_num])

        val_data = pd.concat(val_data)

        #convert to np array with required features. 
        TX = np.array(tr_data[features])
        ty = np.array(tr_data.laughter_value)

        VX = np.array(val_data[features])
        vy = np.array(val_data.laughter_value)

        tree_depth = [3,4,5,6,7,8,9,10]
        t_Accuracies = {}

        #fold = 0
        for t in tree_depth:
            #print('Tree : ', t)
            #fold = fold + 1
            #set the hyper parameters of model
            model = xgb.XGBClassifier(objective='multi:softmax',max_depth=t)

            #fit the model
            model.fit(TX, ty )

            pred = model.predict(VX)

            accuracy = accuracy_score(pred,vy)
            #print(' t : ',t)
            #print(' Accuracy_Score : ', accuracy )

            if t in t_Accuracies:
                t_Accuracies[t].append(accuracy)
            else:
                t_Accuracies[t] = [accuracy]


    t_max_accuracy = -1
    max_accuracy = 0

    for t in t_Accuracies:
        accuracy = np.mean(t_Accuracies[t])

        if max_accuracy < accuracy:
            t_max_accuracy = t
            max_accuracy = accuracy

    #print("t : ", t_max_accuracy)
    #print("max accuracy : " , max_accuracy)


    #print("*"*20)

    TX  = np.array(train_data[features])
    ty = np.array(train_data.laughter_value)

    test_X = np.array(test_data[features])
    test_y = np.array(test_data.laughter_value)

    model = xgb.XGBClassifier(objective='multi:softmax',max_depth=t)

    #fit the model
    model.fit(TX, ty )

    pred = model.predict(test_X)

    accuracy = accuracy_score(pred,test_y)
    #print('t : ',t)
    #print('Accuracy_Score : ', accuracy )

    #print('--'*10)
    
    return [pred,accuracy]       


def main():

	opensmile_std_input_df = pd.read_excel("./Audio/open_face_accoustic_std.xlsx")
	opensmile_mean_input_df = pd.read_excel("./Audio/open_face_accoustic_means.xlsx")
	transcript_input_df = pd.read_csv("./Transcript/segment_annotation_transcript_features.csv")
	video_max_df = pd.read_csv("./Video/feature_annotation_max.csv")
	video_min_df = pd.read_csv("./Video/feature_annotation_min.csv")
	video_mean_df = pd.read_csv("./Video/feature_annotation_mean.csv")
	video_std_df = pd.read_csv("./Video/feature_annotation_sd.csv")

	num_dict = {}
	artists = list(transcript_input_df.video.unique())

	for artist_num in range(len(artists)):
	    num_dict[artists[artist_num]] = artist_num

	artist_num = []
	for artist in transcript_input_df.video:
	    artist_num.append(num_dict[artist])

	artist_num

	transcript_input_df['video_num'] = artist_num
	opensmile_std_input_df['video_num'] = artist_num
	opensmile_mean_input_df['video_num'] = artist_num
	transcript_input_df['video_num'] = artist_num
	video_max_df['video_num'] = artist_num
	video_min_df['video_num'] = artist_num
	video_mean_df['video_num'] = artist_num
	video_std_df['video_num'] = artist_num

	transcript_features = get_features(transcript_input_df.columns)
	opensmile_std_features = get_features(opensmile_std_input_df.columns)
	opensmile_mean_features = get_features(opensmile_mean_input_df.columns)
	video_max_features = get_features(video_max_df.columns)
	video_min_features = get_features(video_min_df.columns)
	video_mean_features = get_features(video_mean_df.columns)
	video_std_features = get_features(video_std_df.columns)


	late_fusion(opensmile_std_input_df,
           video_std_df,
           transcript_input_df,
           opensmile_std_features,
           video_std_features,
           transcript_features)

if __name__ == "__main__":
	main()
