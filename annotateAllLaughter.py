import fnmatch
import os

output_file=open('laughter_annotation.csv','w')
output_file.write('video,laughter_start,laughter_end,segment_start,segment_end,laughter\n')
for file in os.listdir('Laughter_annotation/.'):
	if fnmatch.fnmatch(file, '*.csv'):
		video_name=(file.split('_'))[0]
		input_file=open('Laughter_annotation/'+file , 'r')
		input_data=((input_file.read()).split("\n"))[1:]
		for row in input_data:
			if row!="":
				data=row.split(',')
				output_file.write(video_name+","+data[0]+","+data[1]+"\n")