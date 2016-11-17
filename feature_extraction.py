import statistics
import fnmatch
import math
import os
import scipy.stats as stats

min_output_file=open('feature_annotation_min.csv','w')
max_output_file=open('feature_annotation_max.csv','w')
mean_output_file=open('feature_annotation_mean.csv','w')
sd_output_file=open('feature_annotation_sd.csv','w')
min_output_file.write('video,laughter_start,laughter_end,laughter_value,start_segment_s,end_segment_s,start_segment_frame,end_segment_frame,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_2_z,pose_Tx,pose_Ty,pose_Tz,pose_Rx,pose_Ry,pose_Rz,p_scale,p_rx,p_ry,p_rz,p_tx,p_ty,AU01_r,AU02_r,AU04_r,AU05_r,AU06_r,AU07_r,AU09_r,AU10_r,AU12_r,AU14_r,AU15_r,AU17_r,AU20_r,AU23_r,AU25_r,AU26_r,AU45_r,AU01_c,AU02_c,AU04_c,AU05_c,AU06_c,AU07_c,AU09_c,AU10_c,AU12_c,AU14_c,AU15_c,AU17_c,AU20_c,AU23_c,AU25_c,AU26_c,AU28_c,AU45_c,eyebrow_raise,frown\n')
max_output_file.write('video,laughter_start,laughter_end,laughter_value,start_segment_s,end_segment_s,start_segment_frame,end_segment_frame,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_2_z,pose_Tx,pose_Ty,pose_Tz,pose_Rx,pose_Ry,pose_Rz,p_scale,p_rx,p_ry,p_rz,p_tx,p_ty,AU01_r,AU02_r,AU04_r,AU05_r,AU06_r,AU07_r,AU09_r,AU10_r,AU12_r,AU14_r,AU15_r,AU17_r,AU20_r,AU23_r,AU25_r,AU26_r,AU45_r,AU01_c,AU02_c,AU04_c,AU05_c,AU06_c,AU07_c,AU09_c,AU10_c,AU12_c,AU14_c,AU15_c,AU17_c,AU20_c,AU23_c,AU25_c,AU26_c,AU28_c,AU45_c,eyebrow_raise,frown\n')
mean_output_file.write('video,laughter_start,laughter_end,laughter_value,start_segment_s,end_segment_s,start_segment_frame,end_segment_frame,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_2_z,pose_Tx,pose_Ty,pose_Tz,pose_Rx,pose_Ry,pose_Rz,p_scale,p_rx,p_ry,p_rz,p_tx,p_ty,AU01_r,AU02_r,AU04_r,AU05_r,AU06_r,AU07_r,AU09_r,AU10_r,AU12_r,AU14_r,AU15_r,AU17_r,AU20_r,AU23_r,AU25_r,AU26_r,AU45_r,AU01_c,AU02_c,AU04_c,AU05_c,AU06_c,AU07_c,AU09_c,AU10_c,AU12_c,AU14_c,AU15_c,AU17_c,AU20_c,AU23_c,AU25_c,AU26_c,AU28_c,AU45_c,eyebrow_raise,frown\n')
sd_output_file.write('video,laughter_start,laughter_end,laughter_value,start_segment_s,end_segment_s,start_segment_frame,end_segment_frame,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_2_z,pose_Tx,pose_Ty,pose_Tz,pose_Rx,pose_Ry,pose_Rz,p_scale,p_rx,p_ry,p_rz,p_tx,p_ty,AU01_r,AU02_r,AU04_r,AU05_r,AU06_r,AU07_r,AU09_r,AU10_r,AU12_r,AU14_r,AU15_r,AU17_r,AU20_r,AU23_r,AU25_r,AU26_r,AU45_r,AU01_c,AU02_c,AU04_c,AU05_c,AU06_c,AU07_c,AU09_c,AU10_c,AU12_c,AU14_c,AU15_c,AU17_c,AU20_c,AU23_c,AU25_c,AU26_c,AU28_c,AU45_c,eyebrow_raise,frown\n')


anova_list=list()
for i in range(0,55):
	anova_list.append(list())
	anova_list[i].append(list()) #min
	anova_list[i][0].append(list())
	anova_list[i][0].append(list())
	anova_list[i][0].append(list())

	anova_list[i].append(list()) #max
	anova_list[i][1].append(list())
	anova_list[i][1].append(list())
	anova_list[i][1].append(list())

	anova_list[i].append(list()) #mean
	anova_list[i][2].append(list())
	anova_list[i][2].append(list())
	anova_list[i][2].append(list())

	anova_list[i].append(list()) #sd
	anova_list[i][3].append(list())
	anova_list[i][3].append(list())
	anova_list[i][3].append(list())

get_video_name_file = open('segment_annotation.csv', 'r')
video_input_data=((get_video_name_file.read()).split("\n"))[1:]
for video_data in video_input_data:
	video_data=video_data.strip()
	if video_data!="":
		video_data_list=video_data.split(",")

		video_name=video_data_list[0].strip()
		print video_name

		start_frame=int(video_data_list[6].strip())
		end_frame=int(video_data_list[7].strip())

		laughter_annotation = int(video_data_list[3].strip())

		feature_list=list()
		for i in range(0,55):
			feature_list.append(list())

		for openface_file in os.listdir('OpenFaceOutput/.'):
			if fnmatch.fnmatch(openface_file, video_name+'_Video*'):
				name_of_input_file=openface_file
				break

		openface_input_file=open('OpenFaceOutput/'+name_of_input_file , 'r')
		inp=(openface_input_file.read().split("\n"))[1:]

		for openface_data in inp:
			openface_data=openface_data.strip()
			if openface_data!="":
				values=openface_data.split(",")

				frame_num = int(values[0].strip())
				
				if frame_num>end_frame:
					break

				if frame_num>=start_frame and frame_num<=end_frame:
					for li in range(4,16):
						feature_list[li-4].append(float(values[li].strip()))
					for li in range(152,158):
						feature_list[li-152+12].append(float(values[li].strip()))
					for li in range(192,227):
						feature_list[li-192+18].append(float(values[li].strip()))

					lx_lower=float(values[16+45].strip())
					ly_lower=float(values[84+45].strip())

					lx_upper=float(values[16+25].strip())
					ly_upper=float(values[84+25].strip())

					rx_lower=float(values[16+38].strip())
					ry_lower=float(values[84+38].strip())

					rx_upper=float(values[16+20].strip())
					ry_upper=float(values[84+20].strip())

					ld=math.sqrt(math.pow((lx_lower-lx_upper),2)+math.pow((ly_lower-ly_upper),2))
					rd=math.sqrt(math.pow((rx_lower-rx_upper),2)+math.pow((ry_lower-ry_upper),2))
					avg=(float(ld)+float(rd))/2

					feature_list[53].append(avg)
					

					lx=float(values[16+23].strip())
					ly=float(values[84+23].strip())

					
					rx=float(values[16+22].strip())
					ry=float(values[84+22].strip())
					frown=math.sqrt(math.pow((lx-rx),2)+math.pow((ly-ry),2))
					feature_list[54].append(frown)
				
		for ri in range(0,8):
			min_output_file.write(video_data_list[ri].strip()+",")
			max_output_file.write(video_data_list[ri].strip()+",")
			mean_output_file.write(video_data_list[ri].strip()+",")
			sd_output_file.write(video_data_list[ri].strip()+",")
		for ri in range(0,54):
			if len(feature_list[ri])>1:
				min_output_file.write(str(min(feature_list[ri]))+",")
				anova_list[ri][0][laughter_annotation].append(min(feature_list[ri]))

				max_output_file.write(str(max(feature_list[ri]))+",")
				anova_list[ri][1][laughter_annotation].append(max(feature_list[ri]))

				mean_output_file.write(str(statistics.mean(feature_list[ri]))+",")
				anova_list[ri][2][laughter_annotation].append(statistics.mean(feature_list[ri]))

				sd_output_file.write(str(statistics.stdev(feature_list[ri]))+",")
				anova_list[ri][3][laughter_annotation].append(statistics.stdev(feature_list[ri]))
			else:
				min_output_file.write(",")
				max_output_file.write(",")
				mean_output_file.write(",")
				sd_output_file.write(",")

		if len(feature_list[54])>1:
			min_output_file.write(str(min(feature_list[54]))+"\n")
			anova_list[54][0][laughter_annotation].append(min(feature_list[54]))

			max_output_file.write(str(max(feature_list[54]))+"\n")
			anova_list[54][1][laughter_annotation].append(max(feature_list[54]))

			mean_output_file.write(str(statistics.mean(feature_list[54]))+"\n")
			anova_list[54][2][laughter_annotation].append(statistics.mean(feature_list[54]))

			sd_output_file.write(str(statistics.stdev(feature_list[54]))+"\n")
			anova_list[54][3][laughter_annotation].append(statistics.stdev(feature_list[54]))

		else:
			min_output_file.write("\n")
			max_output_file.write("\n")
			mean_output_file.write("\n")
			sd_output_file.write("\n")

min_output_file.write(",,,,,,,,")
max_output_file.write(",,,,,,,,")
mean_output_file.write(",,,,,,,,")
sd_output_file.write(",,,,,,,,")
for i in range(0,54):
	min_output_file.write(stats.f_oneway(anova_list[i][0][0],anova_list[i][0][1],anova_list[i][0][2])+",")
	max_output_file.write(stats.f_oneway(anova_list[i][1][0],anova_list[i][1][1],anova_list[i][1][2])+",")
	mean_output_file.write(stats.f_oneway(anova_list[i][2][0],anova_list[i][2][1],anova_list[i][2][2])+",")
	sd_output_file.write(stats.f_oneway(anova_list[i][3][0],anova_list[i][3][1],anova_list[i][3][2])+",")

i=54
min_output_file.write(stats.f_oneway(anova_list[i][0][0],anova_list[i][0][1],anova_list[i][0][2])+"\n")
max_output_file.write(stats.f_oneway(anova_list[i][1][0],anova_list[i][1][1],anova_list[i][1][2])+"\n")
mean_output_file.write(stats.f_oneway(anova_list[i][2][0],anova_list[i][2][1],anova_list[i][2][2])+"\n")
sd_output_file.write(stats.f_oneway(anova_list[i][3][0],anova_list[i][3][1],anova_list[i][3][2])+"\n")
