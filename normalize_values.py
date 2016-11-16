import statistics
import fnmatch
import math
import os

for file in os.listdir('Audio_features/.'):
		if fnmatch.fnmatch(file, '*.txt'):
			name_of_input_file=file
			print name_of_input_file+".....",
			input_file=open('Audio_features/'+name_of_input_file , 'r')
			inp=(input_file.read().split("\n"))[1:]
			pitch=list()
			intensity=list()
			for data in inp:
				if data!="":
					values=data.split(",")
					if "--undefined--" not in values[1]:
						pitch.append(float(values[1]))
					if "--undefined--" not in values[2]:
						intensity.append(float(values[2]))
			output_file=open('Normalized_Audio_features/'+name_of_input_file,'w')
			output_file.write('time,pitch,intensity\n')
			min_pitch=min(pitch)
			max_pitch=max(pitch)
			min_intensity=min(intensity)
			max_intensity=max(intensity)
			for data in inp:
				if data!="":
					values=data.split(",")
					output_file.write(values[0]+",")
					if "--undefined--" not in values[1]:
						output_file.write(str(((float(values[1])-min_pitch)/(max_pitch-min_pitch)))+",")
					else:
						output_file.write("--undefined--,")
					if "--undefined--" not in values[2]:
						output_file.write(str(((float(values[2])-min_intensity)/(max_intensity-min_intensity)))+"\n")
					else:
						output_file.write("--undefined--\n")
			print "DONE"
