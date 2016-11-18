import statistics
import fnmatch
import math
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

	
mean_output_file=open('feature_annotation_mean.csv','r')
sd_output_file=open('feature_annotation_sd.csv','r')

plot_labels="gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_2_z,pose_Tx,pose_Ty,pose_Tz,pose_Rx,pose_Ry,pose_Rz,p_scale,p_rx,p_ry,p_rz,p_tx,p_ty,AU01_r,AU02_r,AU04_r,AU05_r,AU06_r,AU07_r,AU09_r,AU10_r,AU12_r,AU14_r,AU15_r,AU17_r,AU20_r,AU23_r,AU25_r,AU26_r,AU45_r,AU01_c,AU02_c,AU04_c,AU05_c,AU06_c,AU07_c,AU09_c,AU10_c,AU12_c,AU14_c,AU15_c,AU17_c,AU20_c,AU23_c,AU25_c,AU26_c,AU28_c,AU45_c,eyebrow_raise,frown"


p_values=list()
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




input_data=((sd_output_file.read()).splitlines())[1:]

for inputs in input_data:
	new_inputs=inputs.strip()
	if new_inputs!="":
		values=new_inputs.split(",")
		if values[0]=="":
			print "FETCHING P VALUES"
			for li in range(0,55):
				p_values.append(values[li+8].strip())
		else:
			laughter_value=int(values[3].strip())
			for li in range(0,55):
				if (values[li+8].strip())!="":
					anova_list[li][3][laughter_value].append(float(values[li+8].strip()))

print len(plot_labels.split(","))
for i in range(0,55):
	if p_values[i].strip()!="":
		print "\n\n"+str((plot_labels.split(","))[i])+"_standard_deviation"
		print len(anova_list[i][3][0])
		print len(anova_list[i][3][1])
		print len(anova_list[i][3][2])
		data = [anova_list[i][3][0],anova_list[i][3][1],anova_list[i][3][2]]
		labels = ['small','medium','big']
		plt.ylabel(str((plot_labels.split(","))[i])+"_standard_deviation")
		plt.boxplot(data,labels= labels)
		plt.savefig("/Users/harshfatepuria/Documents/sd/"+str((plot_labels.split(","))[i])+"_standard_deviation")
		plt.close()

