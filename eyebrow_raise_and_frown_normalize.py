import statistics
import fnmatch
import math
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

def normalize_list(data_list):
	if min(data_list)==max(data_list):
		return data_list
	min_value=float(min(data_list))
	max_value=float(max(data_list))
	new_data_list=list()
	for item in data_list:
		new_data_list.append((float(item-min_value)/float(max_value-min_value)))
	return new_data_list


for openface_file in os.listdir('OpenFaceOutput/.'):
	if fnmatch.fnmatch(openface_file, '*_Video*'):
		name_of_input_file=openface_file
		print name_of_input_file,".....",
		openface_input_file=open('OpenFaceOutput/'+name_of_input_file , 'r')
		inp=openface_input_file.read().split("\n")

		eyebrow_raise=list()
		frown_measure=list()
		generate_output_file=open('NormalizedOpenFaceOutput/'+name_of_input_file,'w')
		generate_output_file.write(inp[0].strip()+",eyebrow_raise,frown\n")
		inp=inp[1:]
		for openface_data in inp:
			openface_data_new=openface_data.strip()
			if openface_data_new!="":
				values=openface_data_new.split(",")
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
				eyebrow_raise.append(avg)

				lx=float(values[16+23].strip())
				ly=float(values[84+23].strip())

				
				rx=float(values[16+22].strip())
				ry=float(values[84+22].strip())
				frown=math.sqrt(math.pow((lx-rx),2)+math.pow((ly-ry),2))
				frown_measure.append(frown)

		eyebrow_raise=normalize_list(eyebrow_raise)
		frown_measure=normalize_list(frown_measure)
		print len(eyebrow_raise),"     ",len(frown_measure)
		i=0
		for openface_data in inp:
			openface_data_new=openface_data.strip()
			if openface_data_new != "":
				generate_output_file.write(openface_data_new+","+str(eyebrow_raise[i])+","+str(frown_measure[i])+"\n")
				i=i+1

