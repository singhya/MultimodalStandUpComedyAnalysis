import pandas
import math
import numpy

def preprocessData(file, extra, ext, cols, type, frames):
    annotation_df = pandas.read_csv('../../Laughter_annotation/segment_annotation.csv', names=['video','laughter_start','laughter_end','laughter_value','start_segment_s','end_segment_s','start_segment_frame','end_segment_frame'])
    file_name = ''
    file_df = ''
    last_end_segment_time = 0
    annotation_df = annotation_df[1:]
    output_mean = []
    output_std = []
    for index, row in annotation_df.iterrows():
        file_under_consideration = row['video']
        if(file_under_consideration!=file_name):
            file_name = file_under_consideration
            last_end_segment_time = 0
            file_df = pandas.read_csv(file+file_name+extra+ext,
                                  names=cols)
        if(type=='audio'):
            segment_data_frame = file_df[(file_df['frameTime'].convert_objects(convert_numeric=True) > float(last_end_segment_time)) & (file_df['frameTime'].convert_objects(convert_numeric=True) <= float(row['laughter_start']))]
        elif(type=='video'):
            segment_data_frame = file_df[
                (file_df['timestamp'].convert_objects(convert_numeric=True) >= float(last_end_segment_time)) & (
                file_df['timestamp'].convert_objects(convert_numeric=True) < float(row['laughter_start']))]
        print(segment_data_frame.shape)
        segment_mean = []
        segment_std = []
        for g, mini_segment_data_frame in segment_data_frame.groupby(numpy.arange(len(segment_data_frame)) // (math.ceil(len(segment_data_frame)/frames))):
            segment_mean = segment_mean + (mini_segment_data_frame.convert_objects(convert_numeric=True).mean(axis=0).values.T.tolist())
            segment_std = segment_std + (mini_segment_data_frame.convert_objects(convert_numeric=True).std(axis=0).values.T.tolist())
        output_mean.append(segment_mean)
        output_std.append(segment_std)
        print(len(output_mean))
        last_end_segment_time = row['laughter_end']
        #if(index == 4):
        #    break
    A = numpy.array(numpy.repeat(range(0,int(frames)), len(cols)))
    B = numpy.array(cols * int(frames))
    output_mean = pandas.DataFrame(data=output_mean, columns=pandas.MultiIndex.from_tuples(zip(A,B), names=['first', 'second']))
    output_std = pandas.DataFrame(data=output_std, columns=pandas.MultiIndex.from_tuples(zip(A,B), names=['first', 'second']))
    return output_mean,output_std

def earlyFusedData(audioPath, audioExtra, audioExt, audioCols, videoPath, vidoeExtra, videoExt, videoCols , frames):
    annotation_df = pandas.read_csv('../../Laughter_annotation/segment_annotation.csv',
                                    names=['video', 'laughter_start', 'laughter_end', 'laughter_value',
                                           'start_segment_s', 'end_segment_s', 'start_segment_frame',
                                           'end_segment_frame'])
    cols = audioCols+videoCols
    file_name = ''
    audio_file_df = ''
    video_file_df = ''
    last_end_segment_time = 0
    annotation_df = annotation_df[1:]
    output_mean = []
    output_std = []
    for index, row in annotation_df.iterrows():
        file_under_consideration = row['video']
        if (file_under_consideration != file_name):
            file_name = file_under_consideration
            last_end_segment_time = 0
            audio_file_df = pandas.read_csv(audioPath + file_name + audioExtra + audioExt,
                                      names=audioCols)
            video_file_df = pandas.read_csv(videoPath + file_name + vidoeExtra + videoExt,
                                      names=videoCols)
        #if(float(last_end_segment_time)<=float(row['laughter_start'])):
        audio_segment_data_frame = audio_file_df[
            (audio_file_df['frameTime'].convert_objects(convert_numeric=True) > float(last_end_segment_time)) & (
                audio_file_df['frameTime'].convert_objects(convert_numeric=True) <= float(row['laughter_start']))]
        video_segment_data_frame = video_file_df[
            (video_file_df['timestamp'].convert_objects(convert_numeric=True) >= float(last_end_segment_time)) & (
                video_file_df['timestamp'].convert_objects(convert_numeric=True) < float(row['laughter_start']))]
        #else:
        #    audio_segment_data_frame = audio_file_df[
        #        (audio_file_df['frameTime'].convert_objects(convert_numeric=True) <= float(row['laughter_start']))]
        #    video_segment_data_frame = video_file_df[
        #        (video_file_df['timestamp'].convert_objects(convert_numeric=True) <= float(row['laughter_start']))]
        segment_mean = []
        segment_std = []
        audio_frames_mean = []
        audio_frames_std = []
        video_frames_mean = []
        video_frames_std = []
        for g, mini_segment_data_frame in audio_segment_data_frame.groupby(
                        numpy.arange(len(audio_segment_data_frame)) // (math.ceil(len(audio_segment_data_frame) / frames))):
            audio_frames_mean.append(mini_segment_data_frame.convert_objects(convert_numeric=True).mean(axis=0).values.T.tolist())
            audio_frames_std.append(mini_segment_data_frame.convert_objects(convert_numeric=True).std(axis=0).values.T.tolist())
        for g, mini_segment_data_frame in video_segment_data_frame.groupby(
                        numpy.arange(len(video_segment_data_frame)) // (math.ceil(len(video_segment_data_frame) / frames))):
            video_frames_mean.append(mini_segment_data_frame.convert_objects(convert_numeric=True).mean(axis=0).values.T.tolist())
            video_frames_std.append(mini_segment_data_frame.convert_objects(convert_numeric=True).std(axis=0).values.T.tolist())
        while(len(audio_frames_mean)<int(frames)):
            audio_frames_mean.append(numpy.zeros(len(audio_cols)).tolist())
            audio_frames_std.append(numpy.zeros(len(audio_cols)).tolist())
        while(len(video_frames_mean)<int(frames)):
            video_frames_mean.append(numpy.zeros(len(video_cols)).tolist())
            video_frames_std.append(numpy.zeros(len(video_cols)).tolist())
        for i in range(0,int(frames)):
            segment_mean = segment_mean + audio_frames_mean[i] + video_frames_mean[i]
            segment_std = segment_std + audio_frames_std[i] + video_frames_std[i]
        output_mean.append(segment_mean)
        if(sum(segment_std)==0):
            print("Found")
        output_std.append(segment_std)
        print(index)
        last_end_segment_time = row['laughter_end']
    A = numpy.array(numpy.repeat(range(0, int(frames)), len(cols)))
    B = numpy.array(cols * int(frames))
    output_mean = pandas.DataFrame(data=output_mean,
                                   columns=pandas.MultiIndex.from_tuples(zip(A, B), names=['first', 'second']))
    output_std = pandas.DataFrame(data=output_std,
                                  columns=pandas.MultiIndex.from_tuples(zip(A, B), names=['first', 'second']))
    return output_mean, output_std
audio_cols = ['frameTime','pcm_fftMag_mfcc[0]', 'pcm_fftMag_mfcc[1]', 'pcm_fftMag_mfcc[2]','pcm_fftMag_mfcc[3]', 'pcm_fftMag_mfcc[4]', 'pcm_fftMag_mfcc[5]','pcm_fftMag_mfcc[6]', 'pcm_fftMag_mfcc[7]', 'pcm_fftMag_mfcc[8]','pcm_fftMag_mfcc[9]', 'pcm_fftMag_mfcc[10]', 'pcm_fftMag_mfcc[11]','pcm_fftMag_mfcc[12]', 'pcm_fftMag_mfcc_de[0]', 'pcm_fftMag_mfcc_de[1]','pcm_fftMag_mfcc_de[5]', 'pcm_fftMag_mfcc_de[6]', 'pcm_fftMag_mfcc_de[7]','pcm_fftMag_mfcc_de[2]', 'pcm_fftMag_mfcc_de[3]', 'pcm_fftMag_mfcc_de[4]','pcm_fftMag_mfcc_de[8]', 'pcm_fftMag_mfcc_de[9]', 'pcm_fftMag_mfcc_de[10]','pcm_fftMag_mfcc_de[11]', 'pcm_fftMag_mfcc_de[12]', 'pcm_fftMag_mfcc_de_de[0]','pcm_fftMag_mfcc_de_de[1]', 'pcm_fftMag_mfcc_de_de[2]','pcm_fftMag_mfcc_de_de[3]', 'pcm_fftMag_mfcc_de_de[4]','pcm_fftMag_mfcc_de_de[5]', 'pcm_fftMag_mfcc_de_de[6]','pcm_fftMag_mfcc_de_de[7]', 'pcm_fftMag_mfcc_de_de[8]','pcm_fftMag_mfcc_de_de[9]', 'pcm_fftMag_mfcc_de_de[10]','pcm_fftMag_mfcc_de_de[11]', 'pcm_fftMag_mfcc_de_de[12]', 'chroma[0]','chroma[1]', 'chroma[2]', 'chroma[3]', 'chroma[4]', 'chroma[5]', 'chroma[6]','chroma[7]', 'chroma[8]', 'chroma[9]', 'chroma[10]', 'chroma[11]','chroma[0].1', 'chroma[1].1', 'chroma[2].1', 'chroma[3].1', 'chroma[4].1','chroma[5].1', 'chroma[6].1', 'chroma[7].1', 'chroma[8].1', 'chroma[9].1','chroma[10].1', 'chroma[11].1', 'pcm_LOGenergy', 'voiceProb_sma', 'F0_sma','pcm_loudness_sma', 'F0final_sma', 'voicingFinalUnclipped_sma','pcm_loudness_sma.1']
video_cols = ['frame','timestamp','confidence','success','gaze_0_x','gaze_0_y','gaze_0_z','gaze_1_x','gaze_1_y','gaze_2_z','pose_Tx','pose_Ty','pose_Tz','pose_Rx','pose_Ry','pose_Rz','x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','x_9','x_10','x_11','x_12','x_13','x_14','x_15','x_16','x_17','x_18','x_19','x_20','x_21','x_22','x_23','x_24','x_25','x_26','x_27','x_28','x_29','x_30','x_31','x_32','x_33','x_34','x_35','x_36','x_37','x_38','x_39','x_40','x_41','x_42','x_43','x_44','x_45','x_46','x_47','x_48','x_49','x_50','x_51','x_52','x_53','x_54','x_55','x_56','x_57','x_58','x_59','x_60','x_61','x_62','x_63','x_64','x_65','x_66','x_67','y_0','y_1','y_2','y_3','y_4','y_5','y_6','y_7','y_8','y_9','y_10','y_11','y_12','y_13','y_14','y_15','y_16','y_17','y_18','y_19','y_20','y_21','y_22','y_23','y_24','y_25','y_26','y_27','y_28','y_29','y_30','y_31','y_32','y_33','y_34','y_35','y_36','y_37','y_38','y_39','y_40','y_41','y_42','y_43','y_44','y_45','y_46','y_47','y_48','y_49','y_50','y_51','y_52','y_53','y_54','y_55','y_56','y_57','y_58','y_59','y_60','y_61','y_62','y_63','y_64','y_65','y_66','y_67','p_scale','p_rx','p_ry','p_rz','p_tx','p_ty','p_0','p_1','p_2','p_3','p_4','p_5','p_6','p_7','p_8','p_9','p_10','p_11','p_12','p_13','p_14','p_15','p_16','p_17','p_18','p_19','p_20','p_21','p_22','p_23','p_24','p_25','p_26','p_27','p_28','p_29','p_30','p_31','p_32','p_33','AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']
mean, std = earlyFusedData('../../OpenSmile/ExtractedFeatures/','_Audio','.csv', audio_cols,'../../OpenFace/ExtractedFeatures/','','.txt', video_cols, 15.0)
mean.to_csv('mean_data.csv', sep=',')
std.to_csv('std_data.csv', sep=',')
'''mean, std = preprocessData('../../OpenFace/ExtractedFeatures/','','.txt', video_cols,'video', 15.0)
mean.to_csv('mean_video_data.csv', sep=',')
std.to_csv('std_video_data.csv', sep=',')
mean, std = preprocessData('../../OpenSmile/ExtractedFeatures/','_Audio','.csv', audio_cols,'audio', 15.0)
mean.to_csv('mean_audio_data.csv', sep=',')
std.to_csv('std_audio_data.csv', sep=',')


print(len(mean))'''