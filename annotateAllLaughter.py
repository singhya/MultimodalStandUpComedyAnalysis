import fnmatch
import os
import statistics

output_file=open('laughter_annotation.csv','w')
output_file.write('video,laughter_start,laughter_end,mean_pitch,sd_pitch,mean_intensity,sd_intensity,laughter_value,start_segment_s,end_segment_s,start_segment_frame,end_segment_frame\n')
for file in os.listdir('Laughter_annotation/.'):
	if fnmatch.fnmatch(file, '*.csv'):
		video_name=(file.split('_'))[0]
		print video_name+".....",
		input_file=open('Laughter_annotation/'+file , 'r')
		input_data=((input_file.read()).splitlines())
		input_data=input_data[1:]
		print len(input_data),"     ",
		for audio_file in os.listdir('Normalized_Audio_features/.'):
			if fnmatch.fnmatch(audio_file, video_name+'_Audio.txt'):
				name_of_input_file=audio_file
				print name_of_input_file+".....",
				break

		audio_feature_input_file=open('Normalized_Audio_features/'+name_of_input_file , 'r')
		inp=(audio_feature_input_file.read().split("\n"))[1:]
		print len(inp),"   ",
		lineNumber=1
		old_laughter_start_segment_s=0
		old_laughter_end_segment_s=0
		new_start_segment_s=0
		new_end_segment_s=0
		for row in input_data:
			if row!="":
				row=row.strip()
				data=row.split(',')

				if lineNumber==1:
					new_start_segment_s=0
					new_end_segment_s=float(data[0])
					lineNumber=2
				else:
					new_start_segment_s=old_laughter_end_segment_s
					new_end_segment_s=float(data[0])

				old_laughter_start_segment_s=float(data[0])
				old_laughter_end_segment_s=float(data[1])
					
				pitch = list()
				intensity = list()
				for audio_data in inp:
					audio_data=audio_data.strip()
					if audio_data != "":
						values = audio_data.split(",")
						time_s=float(values[0])
						if time_s>=float(data[0]) and time_s<=data[1]:
							if "--undefined--" not in values[1]:
								pitch.append(float(values[1]))
							if "--undefined--" not in values[2]:
								intensity.append(float(values[2]))
						elif time_s>data[1]:
							break
				if len(pitch)>2 and len(intensity)>2:
					output_file.write(video_name+","+str(data[0])+","+str(data[1])+","+str(statistics.mean(pitch))+","+str(statistics.stdev(pitch))+","+str(statistics.mean(intensity))+","+str(statistics.stdev(intensity))+","+str(statistics.mean(intensity)*statistics.mean(pitch))+","+str(new_start_segment_s)+","+str(new_end_segment_s)+","+str(int(30*new_start_segment_s)+1)+","+str(int(30*new_end_segment_s))+"\n")
				else:
					output_file.write(video_name+","+str(data[0])+","+str(data[1])+"\n")
		print "DONE"
							